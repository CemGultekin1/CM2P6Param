import itertools
from utils.paths import coarse_graining_projection_weights_path, inverse_coarse_graining_weights_path
import xarray as xr
import os
import matplotlib.pyplot as plt
from data.load import load_xr_dataset
import numpy as np
import gcm_filters
import torch.nn as nn
import torch

def save_filter_inversion(sigma):
    args = f'--sigma {sigma} --mode data'.split()

    ds = load_xr_dataset(args)


    def compute_area(lat):
        dlat = lat[1:] - lat[:-1]
        area_lat = dlat.reshape([-1,1])
        area_lat = np.concatenate([area_lat,area_lat[-1:]],axis = 0)
        return area_lat
    area_lat = compute_area(ds.ulat.values)
    area_lon = compute_area(ds.ulon.values)

    filter_scale = sigma/2*np.sqrt(12)
    dx_min = 1
    specs = {
        'filter_scale': filter_scale,
        'dx_min': dx_min,
        'grid_type': gcm_filters.GridType.REGULAR,
        'filter_shape':gcm_filters.FilterShape.GAUSSIAN,
    }
    gaussian = gcm_filters.Filter(**specs,)#grid_vars = {'wet_mask':wetmask})


    coarsen_specs = dict(boundary = "trim")
    def collect_forward(area,):
        ny = len(area)
        x = xr.DataArray(data = np.zeros((ny,1)),\
            dims=["lat",'lon'],\
            coords = dict(lat = np.arange(ny),\
                lon = np.arange(1)))
        dA = xr.DataArray(data = area,\
            dims=["lat",'lon'],\
            coords = dict(lat = np.arange(ny),\
                lon = np.arange(1)))
        dAbar = gaussian.apply(dA,dims=["lat","lon"])
        xhats = []
        for i in range(ny):
            x.data = x.data*0
            x.data[i] = 1
            xhat = gaussian.apply(x*dA,dims=["lat","lon"])/dAbar
            xhat = xhat.coarsen(lat = sigma,**coarsen_specs).mean()
            xhats.append(xhat.data[:,0])
        xhats = np.stack(xhats,axis = 1)
        return xhats
    lat = ds.ulat.values
    clat = ds.coarsen(ulat = sigma,**coarsen_specs).mean().ulat.values

    lon = ds.ulon.values
    clon = ds.coarsen(ulon = sigma,**coarsen_specs).mean().ulon.values

    yhats = collect_forward(area_lat)
    xhats = collect_forward(area_lon)

    u,s,vh_ = np.linalg.svd(yhats,full_matrices = False)
    pseudoinv_lat = u@np.diag(1/s)@vh_
    latproj = vh_

    u,s,vh = np.linalg.svd(xhats,full_matrices = False)
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

    path = inverse_coarse_graining_weights_path(sigma)
    filters.to_netcdf(path = path)
    path = coarse_graining_projection_weights_path(sigma)
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
    proj.to_netcdf(path = path)
def forward_difference(x:xr.DataArray,field):
        dx = x.diff(field)/x[field].diff(field)
        f0 = x[field][0]
        dx = dx.pad({field : (1,0)},constant_values = np.nan)
        dxf = dx[field].values
        dxf[0] = f0
        dx[field] = dxf
        return dx
def single_component(f,k,i,j,x,field='lat',hres = True):
    if hres:
        A = f[f"inv_{field}"][i,:]
        B = f[f"inv_{field}"][j,:]
        DB = f[f"diff_inv_{field}"][j,:]
        C = f[f"forward_{field}"][k,:]
        return np.sum(A*B*C),np.sum(A*DB*C)

    E = 0

    if i==k:
        x = x*0
        x[j] = 1.
        dx = forward_difference(x,field)
        DE = dx[k]
    else:
        DE = 0
    if k==i and k==j:
        E = 1
    return E,DE

def lat_spec_weights(filters,lati,span):
    nlat,nlon = len(filters.clat),len(filters.clon)

    xlat = xr.DataArray(
        data = np.zeros(nlat),
        dims = ["lat"],
        coords = dict(lat = (["lat"],filters.clat.data))
    )

    xlon = xr.DataArray(
        data = np.zeros(nlon),
        dims = ["lon"],
        coords = dict(lon =(["lon"],filters.clon.data))
    )

    rfield = 2*span + 1
    def fun(latii,x,**kwargs):
        wlat = np.zeros((rfield,rfield))
        dwlat = np.zeros((rfield,rfield))
        for i,j in itertools.product(np.arange(rfield),np.arange(rfield)):
            ii = i - span + latii
            jj = j - span + latii
            wlat[i,j],dwlat[i,j] = single_component(filters,latii,ii,jj,x,**kwargs)
        return wlat,dwlat
    hwlat,hdwlat = fun(lati,xlat,field = 'lat',hres = True)
    hwlon,hdwlon = fun(nlon//2,xlon,field = 'lon',hres = True)

    lwlat,ldwlat = fun(lati,xlat,field = 'lat',hres = False)
    lwlon,ldwlon = fun(nlon//2,xlon,field = 'lon',hres = False)
    return hwlat,hdwlat,hwlon,hdwlon,lwlat,ldwlat,lwlon,ldwlon,

def visualize_weights(sigma):
    path = inverse_coarse_graining_weights_path(sigma)
    filters = xr.open_dataset(path)
    diff_inv_lon = forward_difference(filters.inv_lon,"lon")
    diff_inv_lat = forward_difference(filters.inv_lat,"lat")
    filters =  filters.assign(diff_inv_lon = diff_inv_lon,diff_inv_lat = diff_inv_lat)
    span = 11
    rfield = 2*span + 1

    lati = np.argmin(np.abs(filters.clat.values))
    print('nclat = ',len(filters.clat.values),' lati = ',lati)
    lati = 500

    hwlat,hdwlat,hwlon,hdwlon,lwlat,ldwlat,lwlon,ldwlon, = lat_spec_weights(filters,lati,span)

    # def locality_estimate(x):
    #     n = x.shape[0]//2
    #     loc = np.zeros(n)
    #     for i,j in itertools.product(range(x.shape[0]),range(x.shape[1])):
    #         m = np.maximum(np.abs(i-n),np.abs(j-n))
    #         loc[:m] += x[i,j]**2
    #     return loc/loc[0]
    # fig,axs = plt.subplots(2,2,figsize=(15,15))
    # outs = (hwlat,hdwlat,hwlon,hdwlon)
    # for i,out in enumerate(outs):
    #     ic = i%2
    #     ir = i//2
    #     x = locality_estimate(out)
    #     L = np.where(x>1e-3)[0][-1]
    #     axs[ir,ic].semilogy(x)
    #     axs[ir,ic].set_title(L)
    # fig.savefig('dummy.png')
    # return

    def get_id_conv(span):
        conv = nn.Conv2d(1,(2*span+1)**2,2*span+1,bias = False)
        conv.weight.data = 0*conv.weight.data
        for i,j in itertools.product(np.arange(rfield),np.arange(rfield)):
            k = i*rfield + j
            conv.weight.data[k,0,i,j] = 1
        return conv
    def fill_weights_to_conv(span,hw0,hw1,lw0,lw1):
        conv = nn.Conv2d(1,rfield**2,2*span+1,bias = False)
        conv.weight.data = 0*conv.weight.data
        for i0,j0,i1,j1 in itertools.product(range(rfield),range(rfield),range(rfield),range(rfield)):
            k = i0*rfield + j0
            conv.weight.data[k,0,i1,j1] = hw0[i0,i1]*hw1[j0,j1] - lw0[i0,i1]*lw1[j0,j1]
        return conv

    conv_id = get_id_conv(span)
    conv_dwlat = fill_weights_to_conv(span,hdwlat,hwlon,ldwlat,lwlon)
    conv_dwlon = fill_weights_to_conv(span,hwlat,hdwlon,lwlat,ldwlon)
    def lsr_forcing(u,v,T):
        return torch.sum(conv_id(u)*conv_dwlon(T) + conv_id(v)*conv_dwlat(T),dim = 1,keepdim = True)

    def coarse_grain(u,selkwargs):
        u = u.values
        u[u!=u]= 0
        ubar = (filters.forward_lat.values @ u) @ filters.forward_lon.values.T
        ubar = xr.DataArray(
            data = ubar,
            dims = ["clat","clon"],
            coords = dict(
                clat = (["clat"],filters.clat.data),
                clon  = (["clon"],filters.clon.data)
            )
        )
        return ubar.sel(selkwargs)

    def totorch(datarr):
        x = datarr.values
        return torch.from_numpy(x.reshape([1,1,x.shape[0],x.shape[1]])).type(torch.float32)
    def fromtorch(x,clat,clon):
        x = x.detach().numpy().reshape([x.shape[2],x.shape[3]])
        sp = (len(clat) - x.shape[0])//2
        return xr.DataArray(
            data = x,
            dims = ["clat","clon"],
            coords = dict(
                clat = (["clat"],clat[sp:-sp]),
                clon  = (["clon"],clon[sp:-sp])
            )
        )

    args = f'--sigma {sigma}'.split()
    ds = load_xr_dataset(args).isel(time = 0)
    selkwargs = {'clon':slice(-250,-150),'clat':slice(-40,40)}
    dvdx = forward_difference(ds.v,"ulon")
    dvdy = forward_difference(ds.u,"ulat")
    S1 = coarse_grain(ds.u*dvdx,selkwargs) + coarse_grain(ds.v*dvdy,selkwargs)
    S0 = coarse_grain(ds.u,selkwargs) * forward_difference(coarse_grain(ds.v,selkwargs),"clon") + \
        coarse_grain(ds.v,selkwargs) *  forward_difference(coarse_grain(ds.v,selkwargs),"clat")
    Strue = S0 - S1
    ubar = coarse_grain(ds.u,selkwargs)
    vbar = coarse_grain(ds.v,selkwargs)


    clat,clon = ubar.clat.values,ubar.clon.values
    ubar_,vbar_, = (totorch(a) for a in (ubar,vbar,))
    S = fromtorch(lsr_forcing(ubar_,vbar_,vbar_),clat,clon)
    fig,axs = plt.subplots(1,2,figsize = (30,10))

    Strue.sel(clon = slice(*S.clon.values[[0,-1]]),clat = slice(*S.clat.values[[0,-1]])).plot(ax = axs[0])
    S.plot(ax = axs[1])
    fig.savefig('dummy.png')

def compute_weights(filters,lati,):
    span = 11
    rfield = span*2+1

    hwlat,hdwlat,hwlon,hdwlon,lwlat,ldwlat,lwlon,ldwlon, = lat_spec_weights(filters,lati,span)

    def get_id_weights(span):
        conv = np.zeros(((2*span+1)**2,2*span+1,2*span+1))#n.Conv2d(1,(2*span+1)**2,2*span+1,bias = False)
        for i,j in itertools.product(np.arange(rfield),np.arange(rfield)):
            k = i*rfield + j
            conv[k,0,i,j] = 1
        return conv
    def get_conv_weights(span,hw0,hw1,lw0,lw1):
        conv = np.zeros(((2*span+1)**2,2*span+1,2*span+1))#nn.Conv2d(1,rfield**2,2*span+1,bias = False)
        for i0,j0,i1,j1 in itertools.product(range(rfield),range(rfield),range(rfield),range(rfield)):
            k = i0*rfield + j0
            conv[k,i1,j1] =  lw0[i0,i1]*lw1[j0,j1] - hw0[i0,i1]*hw1[j0,j1]
        return conv

    hwlat,hdwlat,hwlon,hdwlon,lwlat,ldwlat,lwlon,ldwlon, = lat_spec_weights(filters,lati,span)
    conv_dwlat = get_conv_weights(span,hdwlat,hwlon,ldwlat,lwlon)
    conv_dwlon = get_conv_weights(span,hwlat,hdwlon,lwlat,ldwlon)
    return conv_dwlat,conv_dwlon
def save_weights(sigma):
    path = inverse_coarse_graining_weights_path(sigma)
    filters = xr.open_dataset(path)
    diff_inv_lon = forward_difference(filters.inv_lon,"lon")
    diff_inv_lat = forward_difference(filters.inv_lat,"lat")
    filters =  filters.assign(diff_inv_lon = diff_inv_lon,diff_inv_lat = diff_inv_lat)
    M = 80
    N = len(filters.clat.values)
    lats = np.arange(N//M)*M + M//2
    latweights = {}
    for i,lati in enumerate(lats):
        if lati>= N:
            break
        latval = filters.clat.values[lati]
        print(latval,i,'/',len(lats),lati,'/',N)
        conv_dwlat,conv_dwlon = compute_weights(filters,lati)
        conv = np.stack([conv_dwlat,conv_dwlon],axis =0 )
        latweights[int(lati)] = (latval,conv.tolist())
    clats = np.stack([val[0] for val in latweights.values()],axis=0)
    weights = np.stack([np.array(val[1]) for val in latweights.values()],axis=0)
    convdlat = weights[:,0]
    convdlon = weights[:,1]

    wshp = convdlat.shape[1:]
    nchan,latkernel,lonkernel = wshp[0],wshp[1],wshp[2]
    def compress(weights,tol = 1e-6):
        weights = weights.reshape(weights.shape[0],-1)
        u,s,vh = np.linalg.svd(weights,full_matrices = False)
        us = u @ np.diag(s)
        s = np.cumsum(s[::-1]**2)[::-1]
        K = np.where(s<tol)[0][0]
        us = us[:,:K]
        vh = vh[:K,:].reshape(K,*wshp)
        print(K)
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
    root = '/scratch/cg3306/climate/saves/lsrp/'
    path = os.path.join(root,f'compressed_conv_weights_{sigma}.nc')
    lsrp.to_netcdf(path)


def main():
    for sigma in range(4,18,4):
        print('sigma:\t',sigma)
        # save_filter_inversion(sigma)
        save_weights(sigma)



if __name__=='__main__':
    main()
