
from data.load import load_xr_dataset
from transforms.coarse_grain import coarse_graining_1d_generator, coarse_graining_2d_generator
from utils.paths import coarse_graining_projection_weights_path, inverse_coarse_graining_weights_path
from scipy import linalg
import xarray as xr
import numpy as np

def save_coarse_grain_inversion(sigma):
    args = f'--sigma {sigma} --mode data'.split()
    ds = load_xr_dataset(args)
    ds = ds.rename({"ulon" : "lon","ulat":"lat"})#.sel(lon = slice(-120,-110),lat = slice(-30,-20))
    selkwargs = {'lon':slice(-180,-120),'lat':slice(-30,30)}
    # ds = ds.sel(**selkwargs)
    ds = ds.fillna(0)
    lat = ds.lat.values
    lon = ds.lon.values
    cg1 = coarse_graining_1d_generator(ds,sigma)
    ny,nx = len(ds.lat),len(ds.lon)

    x = xr.DataArray(data = np.eye(ny),\
            dims=["lat",'lon'],\
            coords = dict(lat = ds.lat.values,\
                lon = ds.lat.values))
    
    xlat = cg1(x,lat = True)
    x = xr.DataArray(data = np.eye(nx),\
            dims=["lat",'lon'],\
            coords = dict(lat = ds.lon.values,\
                lon = ds.lon.values))
    xlon = cg1(x,lon = True)
    clat = xlat.lat.values
    clon = xlon.lon.values
    yhats = xlat.values
    xhats = xlon.values.transpose()

    u,s,vh_ = linalg.svd(yhats,full_matrices = False)
    pseudoinv_lat = u@np.diag(1/s)@vh_
    latproj = vh_

    u,s,vh = linalg.svd(xhats,full_matrices = False)
    pseudoinv_lon = u@np.diag(1/s)@vh
    lonproj = vh

    filters = xr.Dataset(
        data_vars = dict(
            forward_lat = (["clat","lat"],yhats),
            inv_lat = (["clat","lat"],pseudoinv_lat),
            forward_lon = (["clon","lon"],xhats),
            inv_lon = (["clon","lon"],pseudoinv_lon),
            proj_lat = (["clat","lat"],latproj),
            proj_lon = (["clon","lon"],lonproj),
        ),
        coords = dict(
            clat = clat,
            lat = lat,
            clon = clon,
            lon = lon
        )
    )

    filters.to_netcdf(path = inverse_coarse_graining_weights_path(sigma))
    
    proj = xr.Dataset(
        data_vars = dict(
            proj_lat = (["clat","lat"],latproj),
            proj_lon = (["clon","lon"],lonproj)
        ),
        coords = dict(
            clat = clat,
            lat = lat,
            clon = clon,
            lon = lon
        )
    )
    proj.to_netcdf(path = coarse_graining_projection_weights_path(sigma))


def test(sigma):
    from data.load import load_xr_dataset
    path = inverse_coarse_graining_weights_path(sigma)
    filters = xr.open_dataset(path)
    args = f'--sigma {sigma}'.split()
    ds = load_xr_dataset(args).isel(time= 0)
    ds = ds.fillna(0)
    ds = ds.rename({"ulon" : "lon","ulat":"lat"})
    cg2d = coarse_graining_2d_generator(ds,sigma)
    u = ds.u
    
    def carry_coord_multiples(val, field,sigma):
        cs = filters[field].values[::sigma]
        return cs[np.argmin(np.abs(cs - val))]
    lon0 = carry_coord_multiples(-180,'lon',sigma)
    lon1 = carry_coord_multiples(-120,'lon',sigma)
    lat0 = carry_coord_multiples(-30,'lat',sigma)
    lat1 = carry_coord_multiples(30,'lat',sigma)
    selkwargs = {'lon':slice(lon0,lon1),'lat':slice(lat0,lat1)}

    def getbounds(u,):
        return u.lat.values[[0,-1]],u.lon.values[[0,-1]]

    ubar = cg2d(u).sel(**selkwargs)
    

    loccg2d = coarse_graining_2d_generator(ds.sel(**selkwargs),sigma)
    locubar = loccg2d(u.sel(**selkwargs))
    selkwargs1 = {'lon':slice(*locubar.lon.values[[0,-1]]),'lat':slice(*locubar.lat.values[[0,-1]])}

    def project(u):
        uv = u.values.copy()
        uv = filters.proj_lat.values @ uv @ filters.proj_lon.values.T
        uv = filters.proj_lat.values.T @ uv @ filters.proj_lon.values
        return xr.DataArray(
            data =uv,
            dims = ["lat","lon"],
            coords = dict(
                lat = (["lat"],u.lat.values),
                lon = (["lon"],u.lon.values),
            )
        )
    up = project(u)  
    upbar = cg2d(up).sel(**selkwargs1)
    locupbar = loccg2d(up.sel(**selkwargs))
        

    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(2,3,figsize = (30,15))
    ubar.plot(ax = axs[0,0])
    locubar.plot(ax = axs[0,1])
    err = ubar - locubar
    err = err.isel(lat = slice(3,-3),lon = slice(3,-3))
    err.plot(ax = axs[0,2])

    upbar.plot(ax = axs[1,0])
    locupbar.plot(ax = axs[1,1])
    err = ubar - locupbar
    err = err.isel(lat = slice(3,-3),lon = slice(3,-3))
    err.plot(ax = axs[1,2])
    fig.savefig('dummy.png')

def main():
    for sigma in range(4,6,4):
        print(sigma)
        test(sigma)


if __name__=='__main__':
    main()