from transforms.coarse_grain import coarse_graining_1d_generator, coarse_graining_2d_generator
from utils.paths import inverse_coarse_graining_weights_path#coarse_graining_projection_weights_path, 
from scipy import linalg
import xarray as xr
import numpy as np





def coarse_grain_inversion_weights(ugrid,tgrid,sigma,):
    
    data_vars = {}
    coords = {}
    grids = dict(u = ugrid,t = tgrid)
    for ut in "u t".split():
        lon = f"{ut}lon"
        lat = f"{ut}lat"
        clat = f"c{lat}"
        clon = f"c{lon}"
        # selkwargs = {lon:slice(-180,-170),lat:slice(-30,30)}
        # ds = ds.sel(**selkwargs)
        ds = grids[ut]
        cg1 = coarse_graining_1d_generator(ds,sigma,prefix = ut)
        lat_values = ds.lat.values
        lon_values = ds.lon.values
        ny,nx = len(lat_values),len(lon_values)

        x = xr.DataArray(data = np.eye(ny),\
                dims=[lat,lon],\
                coords = {lat : lat_values,\
                    lon : lat_values})

        xlat = cg1(x,lat = True)
        x = xr.DataArray(data = np.eye(nx),\
                dims=[lat,lon],\
                coords = {lat : lon_values,\
                    lon : lon_values})
        xlon = cg1(x,lon = True)


        clatv = xlat[lat].values
        clonv = xlon[lon].values
        yhats = xlat.values
        xhats = xlon.values.transpose()

        u,s,vh_ = linalg.svd(yhats,full_matrices = False)
        pseudoinv_lat = u@np.diag(1/s)@vh_
        latproj = vh_

        u,s,vh = linalg.svd(xhats,full_matrices = False)
        pseudoinv_lon = u@np.diag(1/s)@vh
        lonproj = vh

        data_vars_ = {
            f'{ut}_forward_lat' :  ([clat,lat],yhats),
            f'{ut}_inv_lat' :  ([clat,lat],pseudoinv_lat),
            f'{ut}_forward_lon' : ([clon,lon],xhats),
            f'{ut}_inv_lon' :  ([clon,lon],pseudoinv_lon),
            f'{ut}_proj_lat' : ([clat,lat],latproj),
            f'{ut}_proj_lon' : ([clon,lon],lonproj),
        }
        coords_ = {
            clat : clatv,
            lat : lat_values,
            clon : clonv,
            lon : lon_values
        }
        data_vars = dict(data_vars,**data_vars_)
        coords = dict(coords,**coords_)
    filters = xr.Dataset(
        data_vars = data_vars,
        coords = coords,
    )
    return filters
def coarse_grain_projection(u,filters,prefix = 'u'):
    plat = f'{prefix}_proj_lat'
    plon = f'{prefix}_proj_lon'
    latproj = filters[plat].values
    lonproj = filters[plon].values
    uval = u.values.copy()
    mask = np.isnan(uval)
    uval[mask] = 0
    puval = (latproj@uval)@(lonproj.T)
    puval = (latproj.T@puval)@lonproj
    puval[mask] = np.nan
    u = xr.DataArray(
        data = puval,
        dims = ['lat','lon'],
        coords = dict(
            lat = u.lat.values,
            lon = u.lon.values
        )
    )
    return u
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
        save_coarse_grain_inversion(sigma)
    # for sigma in range(4,6,4):
    #     print(sigma)
    #     test(sigma)


if __name__=='__main__':
    main()
