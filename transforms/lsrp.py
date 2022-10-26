import itertools
from utils.paths import coarse_graining_projection_weights_path, convolutional_lsrp_weights_path, inverse_coarse_graining_weights_path, search_compressed_lsrp_paths
import xarray as xr
import os
import numpy as np
from transforms.subgrid_forcing import forward_difference
import torch.nn as nn
import torch
from utils.slurm import flushed_print

MAXSPAN = 15
def weights2convolution(weights,):
    rfield = weights.shape[-1]
    weights = weights.reshape(-1,1,rfield,rfield)
    conv = nn.Conv2d(1,weights.shape[0],rfield,bias = False)
    conv.weight.data = torch.from_numpy(weights).type(torch.float32)
    return conv

def read_coords(u):
    if 'clat' in u.coords:
        return u.clat.values,u.clon.values
    else:
        return u.lat.values,u.lon.values


def single_component(f,k,i,j,field='lat'):
    i = i % f[f"inv_{field}"].shape[0]
    j = j % f[f"inv_{field}"].shape[0]
    k = k % f[f"forward_{field}"].shape[0]
    A = f[f"inv_{field}"][i,:]
    B = f[f"inv_{field}"][j,:]
    DB = f[f"diff_inv_{field}"][j,:]
    C = f[f"forward_{field}"][k,:]
    return np.sum(A*B*C),np.sum(A*DB*C)

def lat_spec_weights(filters,lati,span):
    _,nlon = len(filters.clat),len(filters.clon)
    rfield = 2*span + 1
    def fun(latii,**kwargs):
        wlat = np.zeros((rfield,rfield))
        dwlat = np.zeros((rfield,rfield))
        for i,j in itertools.product(np.arange(rfield),np.arange(rfield)):
            ii = i - span + latii
            jj = j - span + latii
            wlat[i,j],dwlat[i,j] = single_component(filters,latii,ii,jj,**kwargs)
        return wlat,dwlat

    hwlat,hdwlat = fun(lati,field = 'lat')
    hwlon,hdwlon = fun(nlon//2,field = 'lon')

    return hwlat,hdwlat,hwlon,hdwlon

def compute_weights(filters,lati,span = 10):
    rfield = span*2+1
    def get_id_weights(span):
        conv = np.zeros(((2*span+1)**2,2*span+1,2*span+1))#nn.Conv2d(1,(2*span+1)**2,2*span+1,bias = False)#
        for i,j in itertools.product(np.arange(rfield),np.arange(rfield)):
            k = i*rfield + j
            conv[k,i,j] = 1
        return conv
    def get_conv_weights(span,hw0,hw1,):
        conv = np.zeros(((2*span+1)**2,2*span+1,2*span+1))#nn.Conv2d(1,rfield**2,2*span+1,bias = False)#
        for i0,j0,i1,j1 in itertools.product(range(rfield),range(rfield),range(rfield),range(rfield)):
            k = i0*rfield + j0
            conv[k,i1,j1] =   - hw0[i0,i1]*hw1[j0,j1]
        return conv
    hwlat,hdwlat,hwlon,hdwlon = lat_spec_weights(filters,lati,span)
    conv_dwlat = get_conv_weights(span,hdwlat,hwlon)
    conv_dwlon = get_conv_weights(span,hwlat,hdwlon)
    return get_id_weights(span),conv_dwlat,conv_dwlon

def save_uncompressed_weights(sigma,span):
    path = inverse_coarse_graining_weights_path(sigma)
    filters = xr.open_dataset(path)
    diff_inv_lon = forward_difference(filters.inv_lon,"lon")
    diff_inv_lat = forward_difference(filters.inv_lat,"lat")
    filters =  filters.assign(diff_inv_lon = diff_inv_lon,diff_inv_lat = diff_inv_lat)
    clats = filters.clat.values#[span: - span]
    fullclat = filters.clat.values
    lats = clats
    latweights = {}
    for lati,latval in enumerate(lats):
        flushed_print(latval,lati,'/',len(lats))
        latii = np.argmin(np.abs(fullclat - latval))
        _,conv_dwlat,conv_dwlon = compute_weights(filters,latii,span = span)
        conv = np.stack([conv_dwlat,conv_dwlon],axis =0 )
        latweights[int(lati)] = (latval,conv)
    clats = np.stack([val[0] for val in latweights.values()],axis=0)
    weights = np.stack([val[1] for val in latweights.values()],axis=0)
    # print('weights.shape',weights.shape)
    convdlat = weights[:,0]
    convdlon = weights[:,1]
    wshp = convdlat.shape[1:]
    nchan,latkernel,lonkernel = wshp[0],wshp[1],wshp[2]

    lsrp = xr.Dataset(
        data_vars = dict(
            weights_dlat = (["clat","shiftchan","latkernel","lonkernel"], convdlat),
            weights_dlon = (["clat","shiftchan","latkernel","lonkernel"], convdlon)
        ),
        coords = dict(
            clat = clats,
            shiftchan = np.arange(nchan),
            latkernel = np.arange(latkernel)-latkernel//2,
            lonkernel = np.arange(lonkernel)-lonkernel//2,
        )
    )
    path = convolutional_lsrp_weights_path(sigma)
    lsrp.to_netcdf(path)

def shrink_shift_span(lspan,span):
    lrfield = lspan*2+1
    z = np.zeros((lrfield,lrfield))
    for i,j in itertools.product(np.arange(-lspan,lspan+1),np.arange(-lspan,lspan+1)):
        if i >=-span and j>=-span and i <= span and j <= span:
            z[i + lspan,j+ lspan] = 1
    z = z.reshape([-1])
    I = np.where(z>0)[0]
    return I
def shrink_lsrp(lsrp,span):
    lspan = len(lsrp.latkernel)//2
    lsrp = lsrp.sel(latkernel = slice(-span,span),lonkernel = slice(-span,span))
    I = shrink_shift_span(lspan,span)
    lsrp = lsrp.isel(shiftchan = I)
    lsrp['shiftchan'] = np.arange(len(I))
    return lsrp
def save_compressed_weights(sigma,span):
    path = convolutional_lsrp_weights_path(sigma)
    lsrp = xr.open_dataset(path)
    lsrp = shrink_lsrp(lsrp,span)

    convdlat = lsrp.weights_dlat.values
    convdlon = lsrp.weights_dlon.values
    wshp = convdlat.shape[1:]

    clats = lsrp.clat.values
    nchan = len(lsrp.shiftchan.values)
    latkernel = len(lsrp.latkernel.values)
    lonkernel = len(lsrp.lonkernel.values)

    def compress(weights,tol = 1e-6):
        weights = weights.reshape(weights.shape[0],-1)
        u,s,vh = np.linalg.svd(weights,full_matrices = False)
        us = u @ np.diag(s)
        cums = np.cumsum(s[::-1]**2)[::-1]
        cums = np.sqrt(cums/cums[0])
        K = np.where(cums<tol)[0][0]
        flushed_print('np.log10(s)',np.log10(cums[:10]))
        flushed_print('rank = ',K)
        us = us[:,:K]
        vh = vh[:K,:].reshape(K,*wshp)
        return K,us,vh


    kdlat,lat_dlat,_convdlat = compress(convdlat)
    kdlon,lat_dlon,_convdlon = compress(convdlon)

    lsrp = xr.Dataset(
        data_vars = dict(
            latitude_transfer_dlat = (["clat","ncomp_dlat"], lat_dlat),
            weights_dlat = (["ncomp_dlat","shiftchan","latkernel","lonkernel"], _convdlat),
            latitude_transfer_dlon = (["clat","ncomp_dlon"], lat_dlon),
            weights_dlon = (["ncomp_dlon","shiftchan","latkernel","lonkernel"], _convdlon)
        ),
        coords = dict(
            clat = clats,
            ncomp_dlat = np.arange(kdlat),
            ncomp_dlon = np.arange(kdlon),
            shiftchan = np.arange(nchan),
            latkernel = np.arange(latkernel)-latkernel//2,
            lonkernel = np.arange(lonkernel)-lonkernel//2,
        )
    )
    path = convolutional_lsrp_weights_path(sigma,span = span)
    lsrp.to_netcdf(path)
def get_projection_weights(sigma):
    filename = coarse_graining_projection_weights_path(sigma)
    ds =  xr.open_dataset(filename)
    return ds

def get_compressed_weights(sigma,span):
    _,spns = search_compressed_lsrp_paths(sigma)

    spns = np.array(spns)
    mspns = spns[spns >= span]
    if len(mspns)==0:
        i = np.argmin(np.abs(spns - span))
        span_ = spns[i]
        path = convolutional_lsrp_weights_path(sigma,span = span_)
        lsrp = xr.open_dataset(path)
        return lsrp
    else:
        i = np.argmin(np.abs(mspns - span))
        span_ = mspns[i]
        path = convolutional_lsrp_weights_path(sigma,span = span_)
        lsrp = xr.open_dataset(path)
        return shrink_lsrp(lsrp,span)


def main():
    # sigmas =  [16]#range(4,18,4)
    # for sigma in sigmas:
    #     flushed_print('sigma,span:\t',sigma,MAXSPAN)
    #     save_uncompressed_weights(sigma,MAXSPAN)
    sigmas = [8,12]
    spans = [2,4,8,12,16]
    for sigma,span in itertools.product(sigmas,spans):
        flushed_print('sigma,span:\t',sigma,span)
        save_compressed_weights(sigma,span)

if __name__=='__main__':
    main()
