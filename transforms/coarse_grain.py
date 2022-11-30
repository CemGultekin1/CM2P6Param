from transforms.grids import forward_difference
import xarray as xr
import numpy as np
import gcm_filters as gcm
from scipy.ndimage import gaussian_filter
# import cupy as cp
# import scipy as cp

def hreslres(uvars,ugrid:xr.Dataset,coarse_grain_u,):
    '''
    Takes high resolution U-grid variables in dictionary uvars and T-grid variables in dictionary tvars
    Takes their fine-grid derivatives across latitude and longitude
    Returns the fine-grid objects and their coarse-grid counterparts and their coarse-grid derivatives across latitude and longitude 
    '''
    def subhres_lres(hresdict,grid,cg):
        lres = {x:cg(y) for x,y in hresdict.items()}
        dybar = cg(grid.dy)
        dxbar = cg(grid.dx)

        dlat = {f"dlat_{x}":forward_difference(y,grid.dy,"lat") for x,y in hresdict.items()}
        dlon = {f"dlon_{x}":forward_difference(y,grid.dx,"lon") for x,y in hresdict.items()}
        hres = dict(hresdict,**dlat,**dlon)
        dlat = {f"dlat_{x}":forward_difference(y,dybar,"lat") for x,y in lres.items()}
        dlon = {f"dlon_{x}":forward_difference(y,dxbar,"lon") for x,y in lres.items()}
        lres = dict(lres,**dlat,**dlon)
        return hres,lres

    uhres,ulres = subhres_lres(uvars,ugrid,coarse_grain_u)
    return uhres,ulres
    
def get_gcm_filter(sigma,wet,area,tripolar = False):
    filter_scale = sigma/2*np.sqrt(12)
    dx_min = 1
    if not tripolar:
        gtype = gcm.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED
    else:
        gtype = gcm.GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED
    specs = {
        'filter_scale': filter_scale,
        'dx_min': dx_min,
        'filter_shape':gcm.FilterShape.GAUSSIAN,
        'grid_type':gtype,
        'grid_vars':{'wet_mask': wet,'area': area},
    }
    area_weighted_filter = gcm.Filter(**specs,)

    specs = {
        'filter_scale': filter_scale,
        'dx_min': dx_min,
        'filter_shape':gcm.FilterShape.GAUSSIAN,
        'grid_type':gcm.GridType.REGULAR,
    }
    nonweighted_filter = gcm.Filter(**specs,)
    return area_weighted_filter,nonweighted_filter


def get_scipy_filter(sigma):
    '''
    return gaussian filter that acts on xr.DataArrays
    applicable to 2D variables only
    '''
    class filter:
        def apply(self,x:xr.DataArray,**kwargs):
            xv = x.values.copy()
            mask = xv!=xv
            xv[mask] = 0
            gx = gaussian_filter(xv,sigma = sigma,mode= 'constant',cval = np.nan)
            return xr.DataArray(
                data = gx,
                dims = ["lat","lon"],
                coords = dict(
                    lon = x.lon.values,lat = x.lat.values
                )
            )
    return filter()

def get_1d_scipy_filter(sigma):
    '''
    return gaussian filter that acts on xr.DataArrays
    applicable to 1D variables only
    '''
    class filter:
        def apply(self,x:np.ndarray):
            return gaussian_filter(x,sigma = sigma,mode= 'wrap')#,cval = 0)
    return filter()
# def togpu(wet_mask):
#     wet_mask_gpu = wet_mask.copy()
#     wet_mask_gpu = wet_mask_gpu.map_blocks(cp.asarray)
#     return wet_mask_gpu
# def tocpu(filtered_gpu):
#     filtered_gpu = filtered_gpu.map_blocks(cp.asnumpy)
#     return filtered_gpu
def coarse_graining_2d_generator(grid:xr.Dataset,sigma,wetmask,greedy_coarse_graining = True,gpu = True,**kwargs):
    '''
    given a 2d rectangular grid :grid: and coarse-graining factor :sigma:
    it returns a Callable that coarse-grains
    '''
    # if gpu:
    #     area = togpu(grid.area)
    #     wetmask = togpu(wetmask)
    # else:
    #     area = grid.area
    area = grid.area
    _weighted_gaussian,_nonweighted_gaussian = get_gcm_filter(sigma,wetmask,area,**kwargs)
    # carea = tocpu(_nonweighted_gaussian.apply(area,dims=['lat','lon']))
    carea = _nonweighted_gaussian.apply(area,dims=['lat','lon'])
    def _gaussian_apply(xx:xr.Dataset):
        x = xx.copy()
        x = x.load().compute()
        # if gpu:
        #     y = tocpu(_weighted_gaussian.apply(togpu(x),dims=['lat','lon']))
        # else:
        #     y = _weighted_gaussian.apply(x,dims=['lat','lon'])
        y = _weighted_gaussian.apply(x,dims=['lat','lon'])
        y = y * grid.area / carea
        return y


    barwetmask = wetmask.coarsen(lat=sigma,lon=sigma,boundary="trim").mean()
    def weighted_gaussian(x:xr.DataArray):
        cx = _gaussian_apply(x)
        if greedy_coarse_graining:
            cxw = wetmask*cx.fillna(0)
            cxw = cxw.coarsen(lat=sigma,lon=sigma,boundary="trim").mean()/barwetmask
        else:
            cxw = cx.coarsen(lat=sigma,lon=sigma,boundary="trim").mean()
        return cxw
    return weighted_gaussian


def coarse_graining_1d_generator(grid,sigma,prefix ="u"):
    '''
    given a 2d rectangular grid :grid: and coarse-graining factor :sigma:
    it returns a Callable that coarse-grains across latitude and longitude separately
    This is only possible if the 2D coarse-graining operation is separable. However 
    it is not due to above 65 latitude bipolar grid. But we approximate a separable
    coarse-graining by averaging the grid separations across latitude and longitude separately.
    This describes a close to original separable grid. 
    '''
    gaussian = get_1d_scipy_filter(sigma)
    slat = f"{prefix}lat"
    slon = f"{prefix}lon"

    dy = grid.dy.mean(dim = ['lon']).values.reshape([-1,])
    dx = grid.dx.mean(dim = ['lat']).values.reshape([-1,])

    

    cdy = gaussian.apply(dy)
    cdx = gaussian.apply(dx)

    coarsen_specs = {"boundary":"trim"}
    area = {slat : (dy,cdy), slon : (dx,cdx)}

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
