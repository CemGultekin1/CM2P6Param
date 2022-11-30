from data.load import load_xr_dataset
from transforms.coarse_grain import coarse_graining_1d_generator, coarse_graining_2d_generator, get_gcm_filter
from transforms.coarse_grain_inversion import coarse_grain_forward, coarse_grain_inversion_weights, coarse_grain_invert
from transforms.grids import get_grid_vars, logitudinal_expansion, logitudinal_expansion_dataset
from utils.paths import coarse_graining_projection_weights_path
from utils.slurm import flushed_print
from utils.xarray import land_fill, plot_ds
import xarray as xr
import numpy as np

def get_grid():
    ds,_ = load_xr_dataset('--mode data --depth 0'.split())
    ds = ds.isel(time = 0)
    return get_grid_vars(ds)
def get_fine_sample():
    ds,_ = load_xr_dataset('--mode data --depth 0'.split())
    ds = ds.isel(time = 0)
    ugrid,_ = get_grid_vars(ds)
    u= ds.u.load()
    u = u.rename(ulat = "lat",ulon = "lon")
    u.name = 'u'
    uwetmask = xr.where(np.isnan(u),0,1)
    return u,ugrid,uwetmask
def save_coarse_sample(sigma):
    u,ugrid,uwetmask = get_fine_sample()
    cgu = coarse_graining_2d_generator(ugrid,sigma,uwetmask)
    ubar = cgu(u)
    ubar.to_netcdf('coarse_grained.nc',mode = 'w')
def get_coarse_sample():
    return xr.open_dataset('coarse_grained.nc').__xarray_dataarray_variable__
    

def save_cg_weights(sigma):
    ugrid,tgrid = get_grid()
    projections = coarse_grain_inversion_weights(ugrid,tgrid,sigma)
    print(coarse_graining_projection_weights_path(sigma))
    projections.to_netcdf(coarse_graining_projection_weights_path(sigma),mode = 'w')


def load_projections(sigma):
    return xr.open_dataset(coarse_graining_projection_weights_path(sigma))

def land_fill_projections(sigma,coarse_wetmask,latfrw,lonfrw,latinv,loninv,saveloc,suffix):
    def compute_projection(wet,frwmap,):
        sub_frwmap = frwmap[wet,:]
        q,_ = np.linalg.qr(sub_frwmap.T)
        return q.T
    for i in range(coarse_wetmask.shape[1]):
        print('lon',i,coarse_wetmask.shape[1])
        weti= coarse_wetmask.values[:,i]
        q = compute_projection(weti,latfrw.values)
        # print(latfrw.shape)
        qq = np.ones((latfrw.shape[0],latfrw.shape[1]))*np.nan
        qq[:q.shape[1],:] = q
        dims = [lonfrw.dims[0]] + list(latfrw.dims)
        q = xr.DataArray(
            data = np.stack([qq],axis= 0),
            dims = dims,
            coords = dict(latfrw.coords,**{dims[0]: [lonfrw[dims[0]].values[i]]}),
            name = f'latitudinal_{suffix}'
        )
        q = xr.Dataset(
            data_vars = {q.name: (q.dims,q.values)},
            coords = q.coords
        )
        print(saveloc)
        chk = {k:len(q[k]) for k in list(q.coords)}
        q = q.chunk(chunks=chk)
        q.to_netcdf(saveloc,mode = 'w' if i==0 else 'a')

    return
    
    uhalf0 = u0 @ loninv[:u0.shape[1],:u0.shape[1]*sigma]
    uhalf = []
    for i in range(uhalf0.shape[0]):
        print('lat',i,wetmask.shape[0])
        uhi =uhalf0[i,:]
        if i%sigma == 0:
            weti= wetmask[i//sigma,:]
            q = compute_projection(weti,lonfrw)
            q = q[:len(uhi),:]
        uhi = uhi - q@(q.T@uhi)

def save_land_fill_projections(sigma):
    ubar = get_coarse_sample()
    coarse_wetmask = xr.where(np.isnan(ubar),0,1)
    pj = load_projections(sigma)
    latfrw = pj['u_forward_lat']
    lonfrw = pj['u_forward_lon']

    latinv = pj['u_inv_lat']
    loninv = pj['u_inv_lon']
    saveloc = '/scratch/cg3306/climate/CM2P6PARAM/land_fill_projections.nc'
    suffix = ''
    land_fill_projections(sigma,coarse_wetmask,latfrw,lonfrw,latinv,loninv,saveloc,suffix)
def land_fill_demo(sigma):
    ubar = get_coarse_sample()
    pj = load_projections(sigma)
    latfrw = pj['u_forward_lat'].values
    lonfrw = pj['u_forward_lon'].values

    latinv = pj['u_inv_lat'].values
    loninv = pj['u_inv_lon'].values


    u0 = ubar.fillna(0).values
    

    wetmask = xr.where(np.isnan(ubar),0,1).values

    def compute_projection(wet,frwmap,):
        sub_frwmap = frwmap[wet,:]
        q,_ = np.linalg.qr(sub_frwmap.T)
        return q

    uhalf0 = latinv.T @ u0
    uhalf = []
    for i in range(wetmask.shape[1]):
        print('lon',i,wetmask.shape[1])
        uhi =uhalf0[:,i]
        weti= wetmask[:,i]
        q = compute_projection(weti,latfrw)
        uhi = uhi - q@(q.T@uhi)
        uhalf.append(uhi.reshape([-1,1]))
    u0 = np.concatenate(uhalf, axis = 1)
    
    uhalf0 = u0 @ loninv[:u0.shape[1],:u0.shape[1]*sigma]
    uhalf = []
    for i in range(uhalf0.shape[0]):
        print('lat',i,wetmask.shape[0])
        uhi =uhalf0[i,:]
        if i%sigma == 0:
            weti= wetmask[i//sigma,:]
            q = compute_projection(weti,lonfrw)
            q = q[:len(uhi),:]
        uhi = uhi - q@(q.T@uhi)
        uhalf.append(uhi.reshape([1,-1]))

    u0 = np.concatenate(uhalf, axis = 0)
    u0 = xr.DataArray(
        data = u0,
        dims = ['lat','lon'],
        coords = dict(lat = pj.ulat.values[:u0.shape[0]],\
            lon = pj.ulon.values[:u0.shape[1]])
    )
    plot_ds({'u0':u0},'u0',ncols = 1)

sigma = 12

# save_cg_weights(sigma)
# save_coarse_sample(sigma)
save_land_fill_projections(sigma)