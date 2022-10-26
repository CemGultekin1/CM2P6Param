from transforms.grids import get_grid_vars, get_separated_grid_vars
import xarray as xr
import numpy as np
import gcm_filters
from scipy.ndimage import gaussian_filter

def get_gcm_filter(sigma):
    filter_scale = sigma/2*np.sqrt(12)
    dx_min = 1
    specs = {
        'filter_scale': filter_scale,
        'dx_min': dx_min,
        'grid_type': gcm_filters.GridType.REGULAR,
        'filter_shape':gcm_filters.FilterShape.GAUSSIAN,
    }
    return gcm_filters.Filter(**specs,)

def get_scipy_filter(sigma):
    class filter:
        def apply(self,x:xr.DataArray,**kwargs):
            gx = gaussian_filter(x.values,sigma = sigma,mode= 'constant',cval = np.nan)
            return xr.DataArray(
                data = gx,
                dims = ["lat","lon"],
                coords = dict(
                    lon = x.lon.values,lat = x.lat.values
                )
            )
    return filter()

def get_1d_scipy_filter(sigma):
    class filter:
        def apply(self,x:np.ndarray):
            return gaussian_filter(x,sigma = sigma,mode= 'constant',cval = 0)
    return filter()

def coarse_graining_2d_generator(gridvar:xr.DataArray,sigma,wetmask :bool= False):
    '''
    given a 2d rectangular grid :ugrid: and coarse-graining factor :sigma:
    it returns a Callable that coarse-grains
    '''
    ugrid = get_grid_vars(gridvar)
    gaussian = get_scipy_filter(sigma)
    def _gaussian_apply(xx:xr.Dataset):
        x = xx.copy()
        x = x.load().compute()
        y = x.assign_coords(lat = np.arange(len(x.lat)),lon = np.arange(len(x.lon)))
        y = gaussian.apply(y,dims=['lat','lon'])
        y = y.assign_coords(lon = x.lon,lat = x.lat)
        return y

    def area2d(grid:xr.DataArray,wetmask:bool):
        if not wetmask:
            dA = grid.area
        else:
            dA = grid.area*grid.wetmask.values
             
        dAbar = _gaussian_apply(dA,)
        return dA,dAbar

    dA,dAbar = area2d(ugrid,wetmask)


    def weighted_gaussian(x:xr.DataArray):
        x = _gaussian_apply(dA*x)/dAbar
        x = x.coarsen(lat=sigma,lon=sigma,boundary="trim").mean()
        return x
    return weighted_gaussian


def coarse_graining_1d_generator(gridvar,sigma,prefix ="u"):
    grid = get_separated_grid_vars(gridvar,prefix = prefix)
    gaussian = get_1d_scipy_filter(sigma)
    area_lat = grid.area_lat.values
    area_lon = grid.area_lon.values

    slat = f"{prefix}lat"
    slon = f"{prefix}lon"

    carea_lat = gaussian.apply(area_lat)
    carea_lon = gaussian.apply(area_lon)

    coarsen_specs = {"boundary":"trim"}
    area = {slat : (area_lat,carea_lat), slon : (area_lon,carea_lon)}

    def coarse_grain1d(data,lat = False,lon = False):
        assert lat + lon == 1
        
        coords = {slat : [[slat],data[slat]],slon : [[slon],data[slon]]}
        if not lat:
            field1 = slon
            field2 = slat
            # axis = 0
        else:
            field1 = slat
            field2 = slon
            # axis = 1
        coords[field1][1] = coords[field1][1].coarsen({field1:sigma},**coarsen_specs).mean()
        for key in coords:
            coords[key][1] = coords[key][1].values
            coords[key] =  tuple(coords[key])
        harea,larea = area[field1]

        n2 = len(data[field2])
        xhats = []
        for i in range(n2):
            datai = data.isel({field2 : i}).values
            xhat = gaussian.apply(datai*harea,)/larea
            xhat = xr.DataArray(
                data = xhat,
                dims = [field1],
                coords = {
                    field1 : np.arange(len(xhat))
                }
            )
            cxhat = xhat.coarsen({field1 : sigma},**coarsen_specs).mean()
            xhats.append(np.squeeze(cxhat.values))
        xhats = np.stack(xhats,axis = 0)
        if lat:
            xhats = xhats.transpose()
        xhats = xr.DataArray(data = xhats,\
                dims=[slat,slon],\
                coords = coords)
        return xhats
    return coarse_grain1d
