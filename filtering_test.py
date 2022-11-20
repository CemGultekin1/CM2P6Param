import itertools
import gcm_filters as gcm
import xarray as xr
import matplotlib.pyplot as plt
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from scipy.ndimage import gaussian_filter

def slow_plot_hres(u,grids):
    print('plotting')
    ax = plt.axes(projection = ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.yrotation = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='k',alpha=.4)
    ax.pcolormesh(grids.geolon_t,grids.geolat_t,u,cmap='seismic')
    plt.savefig('dummy.png')

def plot_var(u,grids,border1,name = 'hres',abs_scale = True,cmap = 'seismic',direct_plot = False):
    fig,ax = plt.subplots(1,1,figsize = (16,8))#,subplot_kw={'projection':ccrs.PlateCarree()})
    uval = u.values.copy()
    uval[uval!=uval] = 0
    umax = np.amax(np.abs(uval))
    kwargs = dict()
    if abs_scale:
        kwargs['vmin'] = -umax
        kwargs['vmax'] = umax
    grids1 = grids.sel(**border1)
    u1 = u.sel(**border1)
    import matplotlib
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap.set_bad('black',.4)
    if not direct_plot:
        pc=ax.pcolormesh(u1.xu_ocean,u1.yu_ocean,u1,cmap=cmap,**kwargs)
        # u1.plot(ax = ax,**kwargs,cmap = cmap)
        # gl = ax.gridlines(draw_labels=True)
        # gl.xlabels_top = False
        # gl.ylabels_right = False
        # ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='k',alpha=.4)
        fig.colorbar(pc)
    else:
        u1.plot(ax = ax)
    ax.set_title( u.name +' '+'x'.join([str(a) for a in u.shape]))
    fig.savefig(f'{name}.png')
    print(f'{name}.png')
    plt.close()


def get_gcm_filter(sigma,wet,grids):
    filter_scale = sigma/2*np.sqrt(12)
    dx_min = 1
    specs = {
        'filter_scale': filter_scale,
        'dx_min': dx_min,
        'filter_shape':gcm.FilterShape.GAUSSIAN,
        'grid_type':gcm.GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED,
        'grid_vars':{'wet_mask': wet,'area': grids.area_u},
    }
    return gcm.Filter(**specs,)

def scipy_filter(sigma,u,wetmask,grids):
    weights = grids.area_u
    weighted_u = u*weights
    weighted_u = weighted_u.fillna(0)
    weights = weights.fillna(0)
    wbar = gaussian_filter(weights.values,sigma = sigma/2,mode= 'constant',cval = np.nan)
    ubar = gaussian_filter(weighted_u.values,sigma = sigma/2,mode= 'constant',cval = np.nan)/wbar
    dims = u.dims
    ubar =  xr.DataArray(
        data = ubar,
        dims = dims,
        coords = u.coords,
    )
    ubar = xr.where(wetmask,ubar,np.nan)
    ubar.name = f'{u.name}bar'
    return ubar

def demo(factor,u,grids,name,border1,**landfillkwargs):
    wet = xr.where(np.isnan(u),0,1).compute()
    filter_u = get_gcm_filter(factor,wet,grids)

    print('filtering...')
    
    ubar_gcm = filter_u.apply(u,dims = ['yu_ocean','xu_ocean'])
    print('filtered')
    u.name = 'original'
    ubar_gcm.name = 'gcm output'
    wet.name = 'wetmask'

    ubar_scipy_lfill = scipy_filter(factor,land_fill(u,1,factor*2,**landfillkwargs), wet,grids)
    ubar_scipy = scipy_filter(factor, u, wet,grids)

    plot_var(u,grids,border1,name = f'{name}-original')
    plot_var(ubar_gcm,grids,border1,name = f'{name}-gcm-filtered')
    plot_var(ubar_scipy,grids,border1,name = f'{name}-scipy-filtered')
    plot_var(ubar_scipy_lfill,grids,border1,name = f'{name}-scipy-filtered-lfill')
    # plot_var(err,grids,border1,name = f'{name}-gcm-scipy-difference',abs_scale = False,cmap = 'inferno')

    coarsen_kwargs = dict(xu_ocean = factor,yu_ocean = factor,boundary = 'trim')
    def roll_coarsen(var:xr.DataArray):
        return var.coarsen(**coarsen_kwargs).mean()
    wetbar = roll_coarsen(wet)
    def coarse_grain(ubar,weighted = True):
        if weighted:
            ubar0 = ubar.fillna(0)*wet
            return roll_coarsen(ubar0)/wetbar
        else:
            ubar0 = ubar.fillna(0)
            return roll_coarsen(ubar0)


    cgrids = roll_coarsen(grids)
    cu_scipy = coarse_grain(ubar_scipy,weighted = False)
    cu_scipy_lfill = coarse_grain(ubar_scipy_lfill,weighted = False)
    cu_gcm = coarse_grain(ubar_gcm,weighted = True)
    err = np.log10(np.abs(cu_gcm - cu_scipy ))
    err_lfill = np.log10(np.abs(cu_gcm - cu_scipy_lfill ))
    diff_err_lfill = err - err_lfill

    err.name = 'log10(|gcm - scipy|)'
    err_lfill.name = 'log10(|gcm - scipy_lfill|)'
    diff_err_lfill.name = 'log10(|gcm - scipy|)-log10(|gcm - scipy_lfill|)'
    cu_scipy.name = 'coarse-grained u-scipy'
    cu_scipy_lfill.name = 'coarse-grained u-scipy_lfill'
    cu_gcm.name = 'coarse-grained u-gcm'

    plot_var(cu_gcm,cgrids,border1,name = f'{name}-{cu_gcm.name}',direct_plot=False)
    plot_var(cu_scipy,cgrids,border1,name = f'{name}-{cu_scipy.name}',direct_plot=False)
    plot_var(cu_scipy_lfill,cgrids,border1,name = f'{name}-{cu_scipy_lfill.name}',direct_plot=False)

    plot_var(err,cgrids,border1,name = f'{name}-{err.name}',direct_plot=False,abs_scale = False,cmap = 'inferno')
    plot_var(err_lfill,cgrids,border1,name = f'{name}-{err_lfill.name}',abs_scale = False,direct_plot=False,cmap = 'inferno')
    plot_var(diff_err_lfill,cgrids,border1,name = f'{name}-{diff_err_lfill.name}',direct_plot=False)
    

def coarse_grain_and_save(data,grids,border,border1,factors,name):
    grids = grids.sel(**border)
    u = data.isel(time = 0).u.sel(**border).load()#usurf.load()

    
    root = 'saves/plots/filtering'
    for factor in factors:
        demo(factor,u,grids,f'{root}/factor-{factor}-{name}-u',border1,zero_tendency = True)
        # raise Exception

    T = data.isel(time = 0).temp.load()#surface_temp.load()

    T = xr.DataArray(
        data = T.values,
        dims = ['yu_ocean','xu_ocean'],
        coords = dict(
            yu_ocean = data.yu_ocean.values,
            xu_ocean = data.xu_ocean.values,
        )
    ).sel(**border)

    for factor in factors:
        demo(factor,T,grids,f'{root}/factor-{factor}-{name}-temp',border1,zero_tendency = False)

def land_fill(u_:xr.DataArray,factor,ntimes,zero_tendency = False):
    u = u_.copy()
    for _ in range(ntimes):
        u0 = u.fillna(0).values
        if zero_tendency:
            wetmask = xr.where(np.isnan(u),1,1).values
        else:
            wetmask = xr.where(np.isnan(u),0,1).values
        
        u0bar = gaussian_filter(u0*wetmask,sigma = factor,mode= 'constant',cval = np.nan)
        wetbar = gaussian_filter(wetmask.astype(float),sigma = factor,mode= 'constant',cval = np.nan)
        u0bar = u0bar/wetbar
        u0bar = xr.DataArray(
            data = u0bar,
            dims = u.dims,
            coords = u.coords
        )
        u = xr.where(np.isnan(u),u0bar,u)
    return u
def land_filling_demo(data,border,border1,sigma,**landfillkwargs):
    u = data.sel(**border).isel(time = 0).u
    
    ncol = 2
    nrow = 4
    fig,axs = plt.subplots(nrow,ncol,figsize = (ncol*8,nrow*8))
    for i,(coli,rowi) in enumerate(itertools.product(range(nrow),range(ncol))):
        ax = axs[coli,rowi]
        u = land_fill(u,sigma,**landfillkwargs)
        uval = u.values.copy()
        uval[uval!=uval] = 0
        umax = np.amax(np.abs(uval))
        kwargs = dict()
        kwargs['vmin'] = -umax
        kwargs['vmax'] = umax
        u1 = u.sel(**border1)
        import matplotlib
        cmap = 'seismic'
        cmap = matplotlib.cm.get_cmap(cmap)
        cmap.set_bad('black',.4)
        pc=ax.pcolormesh(u1.xu_ocean,u1.yu_ocean,u1,cmap=cmap,**kwargs)
        fig.colorbar(pc)
        ax.set_title('filling land')
    fig.savefig(f'land_fill_{sigma}.png')
    print(f'land_fill_{sigma}.png')
    plt.close()
# border = dict(xu_ocean = slice(-60,60),yu_ocean = slice(-60,60))
# border1 = dict(xu_ocean = slice(-30,30),yu_ocean = slice(-30,30))
# data = xr.open_zarr("/scratch/zanna/data/cm2.6/beneath_surface.zarr",consolidated = False).sel(st_ocean = 1450,method = 'nearest')
# grids = xr.open_dataset("/scratch/zanna/data/cm2.6/GFDL_CM2_6_grid.nc")
# coarse_grain_and_save(data,grids,border,border1,[4,8,12,16],'lcl')




border = dict()
border1 = dict()
data = xr.open_zarr("/scratch/zanna/data/cm2.6/beneath_surface.zarr",consolidated = False).sel(st_ocean = 1450,method = 'nearest')
grids = xr.open_dataset("/scratch/zanna/data/cm2.6/GFDL_CM2_6_grid.nc")
coarse_grain_and_save(data,grids,border,border1,[4,8,12,16],'glbl')
# for sigma in [1,2,3,4]:
#     land_filling_demo(data,border,border1,sigma)