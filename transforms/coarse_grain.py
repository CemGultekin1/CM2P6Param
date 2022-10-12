import xarray as xr
import numpy as np
import gcm_filters



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

def coarse_graining_2d_generator(ugrid,sigma,):
    '''
    given a 2d rectangular grid :ugrid: and coarse-graining factor :sigma:
    it returns a Callable that coarse-grains
    '''
    gaussian = get_gcm_filter(sigma)
    def _gaussian_apply(xx:xr.Dataset,):
        x = xx.copy()
        x = x.load().compute()
        y = x.assign_coords(lat = np.arange(len(x.lat)),lon = np.arange(len(x.lon)))
        y = gaussian.apply(y,dims=['lat','lon'])
        y = y.assign_coords(lon = x.lon,lat = x.lat)
        return y

    def area2d(grid:xr.Dataset):
        lat = grid.lat.values
        lon = grid.lon.values
        dlat = lat[1:] - lat[:-1]
        dlon = lon[1:] - lon[:-1]
        area = dlat.reshape([-1,1])@dlon.reshape([1,-1])
        area = np.concatenate([area,area[-1:]],axis = 0)
        area = np.concatenate([area,area[:,-1:]],axis = 1)
        dA = xr.DataArray(
            data=area,
            dims=["lat", "lon"],
            coords=dict(
            lon = lon, lat = lat,),
        )
        dAbar = _gaussian_apply(dA,)
        return dA,dAbar

    dA,dAbar = area2d(ugrid)

    def weighted_gaussian(x:xr.DataArray):
        x = _gaussian_apply(dA*x,)/dAbar
        x = x.coarsen(lat=sigma,lon=sigma,boundary="trim").mean()
        return x
    return weighted_gaussian


def coarse_graining_1d_generator(grid,sigma):
    def area1d(lat):
        dlat = lat[1:] - lat[:-1]
        area_lat = np.concatenate([dlat,dlat[-1:]],axis = 0)
        return area_lat

    lat,lon = grid.lat.values,grid.lon.values
    coarsen_specs = dict(boundary = "trim")

    clat = grid.lat.coarsen(lat = sigma,**coarsen_specs).mean()
    clon = grid.lon.coarsen(lon = sigma,**coarsen_specs).mean()

    area_lat = area1d(lat)
    area_lon = area1d(lon)

    ny,nx = len(area_lat),len(area_lon)

    area_lat = xr.DataArray(data = area_lat.reshape([-1,1]),\
        dims=["lat","lon"],\
            coords = dict(lat = np.arange(ny),lon = np.arange(1)))

    area_lon = xr.DataArray(data = area_lon.reshape([1,-1]),\
        dims=["lat","lon"],\
            coords = dict(lon = np.arange(nx),\
                lat = np.arange(1)))

    gaussian = get_gcm_filter(sigma)

    carea_lat = gaussian.apply(area_lat,dims=["lat","lon"])
    carea_lon = gaussian.apply(area_lon,dims=["lat","lon"])


    area = dict(lat = (area_lat,carea_lat),lon = (area_lon,carea_lon))

    def coarse_grain1d(data,lat = False,lon = False):
        assert lat + lon == 1
        latvec =data.lat.values
        lonvec = data.lon.values

        coords = dict(lat = [["lat"],data.lat],lon = [["lon"],data.lon])
        if not lat:
            field1 = "lon"
            field2 = "lat"
            axis = 0
        else:
            field1 = "lat"
            field2 = "lon"
            axis = 1
        coords[field1][1] = coords[field1][1].coarsen({field1:sigma},**coarsen_specs).mean()
        for key in coords:
            coords[key][1] = coords[key][1].values
            coords[key] =  tuple(coords[key])
        harea,larea = area[field1]

        n2 = len(data[field2])
        n1 = len(data[field1])
        data[field1] = np.arange(n1)
        xhats = []
        for i in range(n2):
            datai = data.isel({field2 : [i]})
            datai[field2] = harea[field2]
            xhat = gaussian.apply(datai*harea,dims=["lat","lon"])/larea
            cxhat = xhat.coarsen({field1 : sigma},**coarsen_specs).mean()
            xhats.append(np.squeeze(cxhat.values))
        xhats = np.stack(xhats,axis = 0)
        data["lat"] = latvec
        data["lon"] = lonvec
        if lat:
            xhats = xhats.transpose()
        xhats = xr.DataArray(data = xhats,\
                dims=["lat","lon"],\
                    coords = coords)
        return xhats
    return coarse_grain1d
