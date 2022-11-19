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
        'grid_type':gcm.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
        'grid_vars':{'wet_mask': wet,'area': grids.area_u},
    }
    return gcm.Filter(**specs,)

def scipy_filter(sigma,u,wetmask,grids,wetmasked = False):
    if wetmasked:
        weights = grids.area_u*wetmask
    else:
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

def demo(factor,u,grids,name,border1):
    wet = xr.where(np.isnan(u),0,1).compute()
    filter_u = get_gcm_filter(factor,wet,grids)

    print('filtering...')
    ubar_gcm = filter_u.apply(u,dims = ['yu_ocean','xu_ocean'])
    print('filtered')
    u.name = 'original'
    ubar_gcm.name = 'gcm output'
    wet.name = 'wetmask'

    ubar_scipy = scipy_filter(factor, u, wet,grids,wetmasked = False)

    err = np.log10(np.abs(ubar_gcm- ubar_scipy ))
    err.name = 'log10(|gcm - scipy|) with wetmask'

    plot_var(u,grids,border1,name = f'{name}-original')
    plot_var(ubar_gcm,grids,border1,name = f'{name}-gcm-filtered')
    plot_var(ubar_scipy,grids,border1,name = f'{name}-scipy-filtered')
    plot_var(err,grids,border1,name = f'{name}-gcm-scipy-difference',abs_scale = False,cmap = 'inferno')

    def coarse_grain(ubar):
        ubar0 = ubar.fillna(0)
        return ubar0.coarsen(xu_ocean = factor,yu_ocean = factor,boundary = 'trim').mean()


    cgrids = grids.coarsen(xu_ocean = factor,yu_ocean = factor,boundary = 'trim').mean()
    cu_scipy = coarse_grain(ubar_scipy)
    cu_gcm = coarse_grain(ubar_gcm)
    wet_density = coarse_grain(wet)
    cu_scipy.name = 'coarse-grained u-scipy'
    cu_gcm.name = 'coarse-grained u-gcm'
    wet_density.name = 'wet density'
    plot_var(wet_density,cgrids,border1,name = f'{name}-wet-density',direct_plot=False)

    cutoffs = [0.4,0.5,0.6,0.7]

    
    for co in cutoffs:
        cu_gcm_masked = xr.where(wet_density>co, cu_gcm,np.nan)
        costr = str(co).replace('.','p')
        cu_gcm_masked.name = f'cu_gcm_masked_{costr}'
        plot_var(cu_gcm_masked,cgrids,border1,name = f'{name}-{cu_gcm_masked.name}',direct_plot=False)

def coarse_grain_and_save(data,grids,border,border1,factors,name):
    grids = grids.sel(**border)
    u = data.isel(time = 0).u.sel(**border).load()#usurf.load()

    
    root = 'saves/plots/filtering'
    for factor in factors:
        demo(factor,u,grids,f'{root}/factor-{factor}-{name}-u',border1)
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
        demo(factor,T,grids,f'{root}/factor-{factor}-{name}-temp',border1)





border = dict(xu_ocean = slice(-60,60),yu_ocean = slice(-60,60))
border1 = dict(xu_ocean = slice(-30,30),yu_ocean = slice(-30,30))
data = xr.open_zarr("/scratch/zanna/data/cm2.6/beneath_surface.zarr",consolidated = False).sel(st_ocean = 1450,method = 'nearest')
grids = xr.open_dataset("/scratch/zanna/data/cm2.6/GFDL_CM2_6_grid.nc")
coarse_grain_and_save(data,grids,border,border1,[4,8,12,16],'lcl')




# border = dict()#xu_ocean = slice(-45,45),yu_ocean = slice(-45,45))
# border1 = dict()#xu_ocean = slice(-20,20),yu_ocean = slice(-20,20))
# data = xr.open_zarr("/scratch/zanna/data/cm2.6/beneath_surface.zarr",consolidated = False).sel(st_ocean = 1450,method = 'nearest')
# grids = xr.open_dataset("/scratch/zanna/data/cm2.6/GFDL_CM2_6_grid.nc")
# coarse_grain_and_save(data,grids,border,border1,[4,8,12,16],'glbl')