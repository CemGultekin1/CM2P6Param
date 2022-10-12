import os
from utils.paths import inverse_coarse_graining_weights_path
import xarray as xr

import matplotlib.pyplot as plt
from models.nets.lsrp import ConvolutionalLSRP
import numpy as np
import itertools
from transforms.grids import assign_tgrid, boundary2kwargs, make_divisible_by_grid
from transforms.coarse_grain import coarse_graining_2d_generator
from transforms.lsrp import SPAN, compute_weights
from transforms.subgrid_forcing import forward_difference, subgrid_forcing
from utils.slurm import flushed_print
from utils.xarray import concat

def test_model(sigma,save_true_force = False):
    path = inverse_coarse_graining_weights_path(sigma)

    filters = xr.open_dataset(path)
    diff_inv_lon = forward_difference(filters.inv_lon,"lon")
    diff_inv_lat = forward_difference(filters.inv_lat,"lat")
    filters =  filters.assign(diff_inv_lon = diff_inv_lon,diff_inv_lat = diff_inv_lat)

    def save_true_forcing():
        from data.load import load_xr_dataset
        args = f'--sigma {sigma}'.split()
        ds = load_xr_dataset(args).isel(time = 0)
        ds = assign_tgrid(ds)
        ds = ds.fillna(0)
        u,v,T = ds.u,ds.v,ds.v
        u = u.rename(ulat = "lat",ulon = "lon")
        v = v.rename(ulat = "lat",ulon = "lon")
        T = T.rename(ulat = "lat",ulon = "lon")

        boundary = (-30,30,-180,-120)
        selkwargs = boundary2kwargs(*make_divisible_by_grid(u,sigma,*boundary))
        flushed_print(selkwargs)

        def project(u):
            uv = u.values.copy()
            flushed_print(filters.proj_lat.values.shape,uv.shape,filters.proj_lon.values.shape)
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

        up,vp,Tp= project(u),project(v),project(T)
        u,v,T = u.sel(**selkwargs),v.sel(**selkwargs),T.sel(**selkwargs)
        up,vp,Tp = up.sel(**selkwargs),vp.sel(**selkwargs),Tp.sel(**selkwargs)

        cg2u = coarse_graining_2d_generator(u,sigma)
        cg2t = coarse_graining_2d_generator(T,sigma)


        Strue,_ = subgrid_forcing(u,v,T,cg2u,cg2t)
        Strue = concat(**Strue)
        for key in "u v T".split():
            Strue = Strue.rename({key : f"S{key}"})
        Sptrue,_ = subgrid_forcing(up,vp,Tp,cg2u,cg2t)
        Sptrue = concat(**Sptrue)
        for key in "u v T".split():
            Strue = Strue.assign({f"S{key}p":Sptrue[key]})

        ubar = cg2u(u)
        vbar = cg2u(v)
        Tbar = cg2t(T).interp(lat = ubar.lat.values,lon = ubar.lon.values)

        upbar = cg2u(up)
        vpbar = cg2u(vp)
        Tpbar = cg2t(Tp).interp(lat = ubar.lat.values,lon = ubar.lon.values)


        Strue = Strue.assign(ubar = ubar,vbar = vbar,Tbar = Tbar)
        Strue = Strue.assign(upbar = upbar,vpbar = vpbar,Tpbar = Tpbar)

        root = '/scratch/cg3306/climate/saves/lsrp/'
        path = os.path.join(root,f'true_forcing_example_{sigma}.nc')
        Strue.to_netcdf(path = path)

    def get_true_forcing():
        root = '/scratch/cg3306/climate/saves/lsrp/'
        path = os.path.join(root,f'true_forcing_example_{sigma}.nc')
        return  xr.open_dataset(path)
    if save_true_force:
        save_true_forcing()
        return

    Strue = get_true_forcing()
    ubar = Strue.ubar
    vbar = Strue.vbar
    Tbar = Strue.Tbar


    lsrps = [ConvolutionalLSRP(sigma,ubar.lat.values,span) for span in [11,7,5]]
    SS = [lsrp.forward(ubar,vbar,Tbar)  for lsrp in lsrps]

    Strue = Strue.sel(lon = slice(*SS[0].lon.values[[0,-1]]),lat = slice(*SS[0].lat.values[[0,-1]]))
    title = f'interpolated_{sigma}'
    fig,axs = plt.subplots(11,3,figsize = (40,100))
    for i,key in enumerate("u v T".split()):
        Strue[f"S{key}"].plot(ax = axs[0,i])
        axs[0,i].set_title(f'Strue[f"S{key}"]')
        Strue[f"S{key}p"].plot(ax = axs[1,i])
        axs[1,i].set_title(f'Strue[f"S{key}p"]')

        for j,(S,lsrp) in enumerate(zip(SS,lsrps)):
            jj = 2 + j
            flushed_print(jj)
            S[key].plot(ax = axs[jj,i])
            axs[jj,i].set_title(f'Sconv[{key}] span={lsrp.span}')

        for j,(S,lsrp) in enumerate(zip(SS,lsrps)):
            jj = 2 + 3 +j
            flushed_print(jj)
            err = Strue[f"S{key}p"] - S[key]
            err.plot(ax = axs[jj,i])
            axs[jj,i].set_title(f'compression err span={lsrp.span}')

        for j,(S,lsrp) in enumerate(zip(SS,lsrps)):
            jj = 2 + 3*2 + j
            flushed_print(jj)
            err = Strue[f"S{key}"] - S[key]
            err.plot(ax = axs[jj,i])
            axs[jj,i].set_title(f'parameterization err span={lsrp.span}')
    fig.savefig(f'{title}.png')



    fig,axs = plt.subplots(3,3,figsize = (40,30))
    for i,key in enumerate("u v T".split()):
        Strue[f"{key}bar"].plot(ax = axs[i,0])
        axs[i,0].set_title(f'Strue[{f"{key}bar"}]')
        Strue[f"{key}pbar"].plot(ax = axs[i,1])
        axs[i,1].set_title(f'Strue[f"{key}pbar"]')
        err = Strue[f"{key}bar"] - Strue[f"{key}pbar"]
        err.plot(ax = axs[i,2])
        axs[i,2].set_title(f'Strue[f"{key}bar"] - Strue[f"{key}pbar"]')
    fig.savefig(f'/scratch/cg3306/climate/saves/plots/lsrp/projection_test_{sigma}.png')



def plot_weights(sigma):
    span = SPAN
    lsrpm = ConvolutionalLSRP(sigma,np.arange(span*2+1)-span)
    w0,w1 = lsrpm.single_latitude_weights(0)
    # w0,w1 = w0[2:-2,2:-2,2:-2,2:-2],w1[2:-2,2:-2,2:-2,2:-2]
    def get2x2(w):
        shp = list(w.shape)
        nx = shp[0]*shp[2]
        ny = shp[1]*shp[3]
        W = np.zeros((nx,ny))
        for i,j in itertools.product(range(shp[0]),range(shp[1])):
            subw  = w[i,j]
            ii = slice(i*shp[2],(i+1)*shp[2])
            jj = slice(j*shp[3],(j+1)*shp[3])
            W[ii,jj] = subw
            W[ii.stop-1,:] = np.nan
            W[:,jj.stop-1] = np.nan

        return xr.DataArray(W)


    W0,W1 = get2x2(w0),get2x2(w1)
    fig,axs = plt.subplots(1,2,figsize = (50,20))
    W0.plot(ax = axs[0])
    W1.plot(ax = axs[1])
    specs = dict(fontsize = 30)
    fig.suptitle('$S_T = \overline{v} \overline{\partial_y} \overline{T} - \overline{ v \partial_y T } + \overline{u} \overline{\partial_x} \overline{T} - \overline{ u \partial_x T }$',**specs)
    axs[0].set_title('$\overline{v}  \overline{\partial_y} \overline{T} - \overline{ v \partial_y T }$',**specs)
    axs[1].set_title('$\overline{u}  \overline{\partial_x} \overline{T} - \overline{ u \partial_x T }$ ',**specs)
    for i in range(2):
        axs[i].set_ylabel('latitude',**specs)
        axs[i].set_xlabel('longitude',**specs)
    fig.savefig(f'/scratch/cg3306/climate/saves/plots/lsrp/lsrp_weights_{sigma}.png')
    flushed_print(f'/scratch/cg3306/climate/saves/plots/lsrp/lsrp_weights_{sigma}.png')


def lsrp_weight_interpolation(sigma):
    # save_weights(sigma)
    root = '/scratch/cg3306/climate/saves/lsrp/'
    path = os.path.join(root,f'conv_weights_{sigma}.nc')
    lsrp = xr.open_dataset(path)
    flushed_print(lsrp.weights_dlat.shape)
    clats = lsrp.clat.values
    def compress(w):
        wshp = w.shape
        w = w.reshape(wshp[0],-1)
        u,s,vh = np.linalg.svd(w,full_matrices = False)

        tol = 1e-9
        cs = np.cumsum(s[::-1]**2)[::-1]
        cs = cs/cs[0]
        flushed_print('np.log10(s)',np.log10(s))
        K = np.where(cs<tol**2)[0][0]
        K = len(s)-1
        us = u@np.diag(s)
        us = us[:,:K]
        vh = vh[:K,:].reshape(K,*wshp[1:])
        return us,vh
    le_dlat,cw_dlat = compress(lsrp.weights_dlat.values)
    le_dlon,cw_dlon = compress(lsrp.weights_dlon.values)
    clsrp = xr.Dataset(
        data_vars = dict(
            lt_dlat = (["clat","ncomp_dlat"], le_dlat),
            cw_dlat = (["ncomp_dlat","shiftchan","latkernel","lonkernel"], cw_dlat),
            lt_dlon = (["clat","ncomp_dlon"], le_dlon),
            cw_dlon = (["ncomp_dlon","shiftchan","latkernel","lonkernel"], cw_dlon)
        ),
        coords = dict(
            clat = clats,
            ncomp_dlat = np.arange(le_dlat.shape[1]),
            ncomp_dlon = np.arange(le_dlon.shape[1]),
            shiftchan = lsrp.shiftchan.values,
            latkernel = lsrp.latkernel.values,
            lonkernel = lsrp.lonkernel.values,
        )
    )
    def decompress(latval):
        # clat = clsrp.clat.values
        # i = np.argmin(np.abs(clat - latval))
        # iclsrp = clsrp.isel(clat = [i])
        iclsrp = clsrp.interp(clat = latval)
        def _decompress(key):
            lt = iclsrp[f"lt_d{key}"].values
            cw = iclsrp[f"cw_d{key}"].values
            cwshape = cw.shape
            cw = cw.reshape(cwshape[0],-1)
            w = lt @ cw
            w = w.reshape(*cwshape[1:])
            return w
        wdlat,wdlon = _decompress('lat'),_decompress('lon')
        return wdlat,wdlon


    root = '/scratch/cg3306/climate/saves/lsrp/'
    path = os.path.join(root,f'inv_weights_{sigma}.nc')
    filters = xr.open_dataset(path)
    diff_inv_lon = forward_difference(filters.inv_lon,"lon")
    diff_inv_lat = forward_difference(filters.inv_lat,"lat")
    filters =  filters.assign(diff_inv_lon = diff_inv_lon,diff_inv_lat = diff_inv_lat)
    clats = filters.clat.sel(clat = slice(-15,15)).values
    for lati,latval in enumerate(clats):
        latii = np.argmin(np.abs(filters.clat.values - latval))
        _,dwlat,dwlon = compute_weights(filters,latii)
        cwdlat,cwdlon = decompress(latval)
        flushed_print(latval,np.linalg.norm(dwlat - cwdlat),np.linalg.norm(dwlon - cwdlon))



def main():
    for sigma in range(4,18,4):
        flushed_print('sigma = ',sigma)
        test_model(sigma,save_true_force= True)
        test_model(sigma,save_true_force= False)


if __name__=='__main__':
    main()
