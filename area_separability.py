from data.paths import get_high_res_grid_location
from transforms.grids import get_grid_vars, get_separated_grid_vars
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
def main():
    grid = xr.open_dataset(get_high_res_grid_location())
    grid = grid.rename({'xt_ocean':'tlon','yt_ocean':'tlat'})
    grid = grid.rename({'xu_ocean':'ulon','yu_ocean':'ulat'})
    grid2dt = get_grid_vars(grid.area_t,prefix = 't').rename({'tlon':'ulon','tlat':'ulat',})
    grid2du = get_grid_vars(grid.area_u,prefix = 'u')


    # print(grid2d)
    # print(grid)
    
    vars = [grid2dt.dtlat,grid2du.dulon,grid.dyu,grid.dxu]
    maxvars = np.amax([v.max() for v in vars])
    minvars = np.amin([v.min() for v in vars])
    kwargs = {'vmin':minvars,'vmax':maxvars}
    datakwargs = dict(
        dims = ['ulat','ulon'],\
        coords= dict(ulat = grid.ulat.values,ulon = grid.ulon.values)
    )
    grid['err_dyu'] = xr.DataArray(data = grid.dyu.values - grid2dt.dtlat.values,**datakwargs)
    grid['err_dxu'] = xr.DataArray(data = grid.dxu.values - grid2du.dulon.values,**datakwargs)
    print(np.amax(np.abs(grid.err_dyu.sel(ulat = slice(-1e3,64.999)))))
    print(np.amax(np.abs(grid.err_dxu.sel(ulat = slice(-1e3,64.999)))))
    return

    fig,axs = plt.subplots(4,2,figsize = (40,50))
    grid2dt.dtlat.plot(ax = axs[0,0],**kwargs)
    axs[0,0].set_title('dyu derived from yu_ocean')
    grid2du.dulon.plot(ax = axs[0,1],**kwargs)
    axs[0,1].set_title('dxu derived from xu_ocean')

    grid.dyu.plot(ax = axs[1,0],**kwargs)
    axs[1,0].set_title('original dyu')
    grid.dxu.plot(ax = axs[1,1],**kwargs)
    axs[1,1].set_title('original dxu')

    

    grid.err_dyu.sel(ulat = slice(65.00001,1e3)).plot(ax = axs[2,0])
    axs[2,0].set_title('error dyu above 65 latitude')
    grid.err_dxu.sel(ulat = slice(65.00001,1e3)).plot(ax = axs[2,1])
    axs[2,1].set_title('error dxu above 65 latitude')

    grid.err_dyu.sel(ulat = slice(-1e3,64.999)).plot(ax = axs[3,0])
    axs[3,0].set_title('error dyu below 65 latitude')
    grid.err_dxu.sel(ulat = slice(-1e3,64.999)).plot(ax = axs[3,1])
    axs[3,1].set_title('error dxu below 65 latitude')


    fig.savefig('area_comparison.png')


if __name__ == '__main__':
    main()