from transforms.grids import forward_difference
import xarray as xr
import numpy as np
import gcm_filters
from scipy.ndimage import gaussian_filter


def hreslres(uvars,tvars,ugrid:xr.Dataset,tgrid:xr.Dataset,coarse_grain_u,coarse_grain_t):
    '''
    converts hres :u,v,t: into lres versions by coarse-graining
    including their derivatives across latitude and longitude
    '''
    # flushed_print('ugrid2tgrid(u,v,ugrid,tgrid)')
    # u_t,v_t = ugrid2tgrid(u,v,ugrid,tgrid)
    # if projections is not None:
        # import matplotlib.pyplot as plt
        # def plotsave(u_t,u_t1,name):
        #     fig,axs = plt.subplots(1,2,figsize = (25,10))
        #     u_t.plot(ax = axs[0])
        #     u_t1.plot(ax = axs[1])
        #     fig.savefig(f'{name}.png')
        #     plt.close()
        # flushed_print("coarse_grain_projection(u_t,projections,prefix = 't')")
        # u_t = coarse_grain_projection(u_t,projections,prefix = 't')
        # v_t = coarse_grain_projection(v_t,projections,prefix = 't')
        # T = coarse_grain_projection(T,projections,prefix = 't')
        # u = coarse_grain_projection(u,projections,prefix = 'u')
        # v = coarse_grain_projection(v,projections,prefix = 'u')
        # flushed_print("plotsave(u,u1,'u')")
        # plotsave(u,u1,'u')
        # plotsave(v,v1,'v')
        # plotsave(T,T1,'T')
        # plotsave(u_t,u_t1,'u_t')
        # plotsave(v_t,v_t1,'v_t')
        # u_t = u_t1
        # v_t = v_t1
        # T = T1
        # u = u1
        # v = v1
        # raise Exception
        # import matplotlib.pyplot as plt
        # u.plot()
        # plt.savefig('projected_u.png')
        # plt.close()
    # else:
    #     import matplotlib.pyplot as plt
    #     u.plot()
    #     plt.savefig('u.png')
    #     plt.close()

    # uvars = dict(u=u,v=v)
    # tvars = dict(u=u_t,v=v_t, T = T)
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
    thres,tlres = subhres_lres(tvars,tgrid,coarse_grain_t)
    return uhres,ulres,thres,tlres


    
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
    class filter:
        def apply(self,x:np.ndarray):
            return gaussian_filter(x,sigma = sigma,mode= 'constant',cval = 0)
    return filter()

def coarse_graining_2d_generator(grid:xr.Dataset,sigma,wetmask :bool= False):
    '''
    given a 2d rectangular grid :ugrid: and coarse-graining factor :sigma:
    it returns a Callable that coarse-grains
    '''
    gaussian = get_scipy_filter(sigma)
    def _gaussian_apply(xx:xr.Dataset):
        x = xx.copy()
        x = x.load().compute()
        y = x.assign_coords(lat = np.arange(len(x.lat)),lon = np.arange(len(x.lon)))
        y = gaussian.apply(y,dims=['lat','lon'])
        y = y.assign_coords(lon = x.lon,lat = x.lat)
        return y

    def area2d(wetmask:bool):
        if not wetmask:
            dA = grid.area
        else:
            dA = grid.area*grid.wetmask
             
        dAbar = _gaussian_apply(dA,)
        return dA,dAbar

    dA,dAbar = area2d(wetmask)


    def weighted_gaussian(x:xr.DataArray):
        cx = _gaussian_apply(dA*x)/dAbar
        cx = cx.coarsen(lat=sigma,lon=sigma,boundary="trim").mean()
        return cx
    return weighted_gaussian


def coarse_graining_1d_generator(grid,sigma,prefix ="u"):
    # grid = get_separated_grid_vars(gridvar,prefix = prefix)
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
