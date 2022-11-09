from typing import Callable
from data.vars import get_var_mask_name
import xarray as xr
import torch
import numpy as np
import torch.nn as nn


def plot_ds(ds,imname,ncols = 3):
    import matplotlib.pyplot as plt
    import itertools
    flat_vars = {}

    for key in ds.data_vars.keys():
        u = ds[key]
        if 'tr_depth'in u.dims:
            for i in range(len(u.tr_depth)):
                ui = u.isel(tr_depth = i)
                if np.all(np.isnan(ui.values)):
                    continue
                flat_vars[f'{key}_tr_depth_{i}'] = u.isel(tr_depth = i)
        else:
            flat_vars[key] = u
    vars = list(flat_vars.keys())
    nrows = int(np.ceil(len(vars)/ncols))
    fig,axs = plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*5))
    for z,(i,j) in enumerate(itertools.product(range(nrows),range(ncols))):
        ax = axs[i,j]
        if z >= len(vars):
            continue
        u = flat_vars[vars[z]]
        u.plot(ax = ax)
        ax.set_title(vars[z])
    fig.savefig(imname)
    plt.close()


def tonumpydict(x:xr.Dataset):
    vars = list(x.data_vars)
    data_vars = {}
    for var in vars:
        data_vars[var] = (list(x[var].dims),x[var].values)

    coords = {}
    for c in list(x.coords):
        coords[c] = x[c].values
        if c == 'time':
            coords[c] = np.array(coords[c]).astype(str)
    return data_vars,coords
def fromnumpydict(data_vars,coords):
    for key in data_vars:
        batchnum = data_vars[key][1].shape[0]
        break
    for i in range(batchnum):
        data_vars_ = {}
        coords_ = {}
        for key in data_vars:
            data_vars_[key] = ([t[i] for t in data_vars[key][0]],data_vars[key][1][i].numpy())
        for key in coords:
            if key != 'time':
                coords_[key] = coords[key][i].numpy()
            else:
                coords_[key] =  np.array([coords[key][i]])#datetime.fromisoformat(coords[key][i])])
        ds = xr.Dataset(data_vars = data_vars_,coords = coords_)
        return ds

def fromtorchdict(data_vars,coords,masks,normalize = False,denormalize = False,fillvalue = np.nan):
    ds = fromtorchdict2dataset(data_vars,coords)
    dsmasks = fromtorchdict2dataset(masks,coords)
    ds = mask_dataset(ds,dsmasks,fillvalue = fillvalue)
    if normalize:
        ds = normalize_dataset(ds,denormalize=False)
    elif denormalize:
        ds = normalize_dataset(ds,denormalize=True)
    ds = remove_normalization(ds)
    return drop_unused_coords(ds)
def drop_unused_coords(ds):
    cns = list(ds.coords.keys())
    dims = []
    for key in ds.data_vars.keys():
        dims.extend(list(ds[key].dims))
    dims = np.unique(dims)
    dropcns = [c for c in cns if c not in dims]
    for dcn in dropcns:
        ds = ds.drop(dcn)
    return ds
def fromtorchdict2dataset(data_vars,coords):
    for key in data_vars:
        data_vars[key] = (data_vars[key][0],data_vars[key][1].numpy())
    for key,val in coords.items():
        if isinstance(val,torch.Tensor):
            coords[key] = val.numpy()
    ds = xr.Dataset(data_vars = data_vars,coords = coords)
    return ds

def normalize_dataset(ds,denormalize = False):
    for key in ds.data_vars.keys():
        if 'mean' in key or 'std' in key:
            continue
        mkey = f"{key}_mean"
        skey = f"{key}_std"

        a,b = ds[mkey].values,ds[skey].values

        dims1 = ds[key].dims
        dims0 = ds[mkey].dims
        shp = []
        for d in dims1:
            if d in dims0:
                shp.append(len(ds[d]))
            else:
                shp.append(1)
        a,b = a.reshape(shp),b.reshape(shp)
        if denormalize:
            ds[key] = ds[key] *b + a
        else:
            ds[key] = (ds[key] - a)/b
    return ds
def mask_dataset(ds,maskds,fillvalue = np.nan):
    for key in ds.data_vars.keys():
        mkey = get_var_mask_name(key)
        if mkey in maskds.data_vars:
            ds[key] = xr.where(maskds[mkey],ds[key],fillvalue)
    return ds

def remove_normalization(ds):
    data_vars = {}
    coords = ds.coords
    for key,val in ds.data_vars.items():
        if '_mean' not in key and '_std' not in key:
            data_vars[key] = val
    return xr.Dataset(data_vars = data_vars,coords = coords)

def no_nan_input_mask(u,span,codition:Callable = lambda x:  np.isnan(x),same_size = False)->xr.DataArray:
    '''
    0 for values with a value that satsifies the condition in the its inputs
    1 for no such value in its input
    '''
    mask = xr.where(codition(u), 1, 0)
    if span == 0 :
        return mask
    mask = mask.values
    shp = mask.shape
    sp = span
    shpp = np.array(shp) - 2*sp
    torchmask = torch.from_numpy(mask.reshape([1,1,shp[-2],shp[-1]])).type(torch.float32)
    pool = nn.Conv2d(1,1,2*sp+1,bias = False)
    pool.weight.data = torch.ones(1,1,2*sp+1,2*sp+1).type(torch.float32)/((2*sp+1)**2)
    with torch.no_grad():
        poolmask = pool(torchmask).numpy().reshape(*shpp)
    if same_size:
        mask = np.ones(shp,dtype = float)
        mask[sp:-sp,sp:-sp] = poolmask
        lat = u.lat.values
        lon = u.lon.values
    else:
        mask = poolmask
        lat = u.lat.values[sp:-sp]
        lon = u.lon.values[sp:-sp]
    mask = xr.DataArray(
        data = mask,
        dims = ["lat","lon"],
        coords = dict(
            lat = (["lat"],lat),
            lon = (["lon"],lon)
        )
    )
    return mask


def concat_datasets(x,y):
    for key in y:
        for key_ in y[key]:
            v1 = y[key][key_]
            x[key][key_].append(v1)
    return x


def unpad(fields):
    for key in fields:
        field = fields[key]
        vals,lats,lons = [],[],[]
        for i,(vec,lat,lon)  in enumerate(zip(field['val'],field['lat'],field['lon'])):
            nlat = torch.isnan(lat).sum()
            nlon = torch.isnan(lon).sum()
            vec = vec.numpy()
            lat = lat.numpy()
            lon = lon.numpy()
            if nlat>0:
                vec = vec[:-nlat]
                lat = lat[:-nlat]
            if nlon>0:
                vec = vec[:,-nlon]
                lon = lon[:-nlon]
            assert not np.any(np.isnan(lon))
            assert not np.any(np.isnan(lat))
            vals.append(vec)
            lats.append(lat)
            lons.append(lon)
        field['val'],field['lat'],field['lon'] = vals,lats,lons
        fields[key] = field
    return fields


def numpydict2dataset(outs,time = 0):
    datarrs = {}
    outs = unpad(outs)
    for key in outs:
        vals = outs[key]['val']
        lons = outs[key]['lon']
        lats = outs[key]['lat']
        datarrs[key] = []
        for i in range(len(vals)):
            val = vals[i]
            lat = lats[i]
            lon = lons[i]

            datarr = xr.DataArray(
                    data = np.stack([val],axis =0 ),
                    dims = ["time","lat","lon"],
                    coords = dict(
                        time = [time],lat = lat,lon = lon,
                    ),
                    name = key
                )
            datarrs[key].append(datarr)

    keys = list(outs.keys())
    merged_dataarrs = {}
    for key in keys:
        out = datarrs[key][0]
        for i in range(1,len(datarrs[key])):
            out = out.combine_first(datarrs[key][i])
        merged_dataarrs[key] = out
    return xr.Dataset(data_vars = merged_dataarrs)



def concat(**kwargs):
    data_vars = dict()
    for name,var in kwargs.items():
        data_vars[name] = (["lat","lon"],var.data)
    coords = dict(
        lon = (["lon"],var.lon.values),
        lat = (["lat"],var.lat.values),
    )
    return xr.Dataset(data_vars = data_vars,coords = coords)


def totorch(*args,**kwargs):
    return (_totorch(arg,**kwargs) for arg in args)

def _totorch(datarr:xr.DataArray,leave_nan = False):
    if not leave_nan:
        # x = datarr.interpolate_na(dim = "lat").interpolate_na(dim = "lon").values
        x = datarr.fillna(0).values
    else:
        x = datarr.values
    return torch.from_numpy(x.reshape([1,1,x.shape[0],x.shape[1]])).type(torch.float32)
def fromnumpy(x,clat,clon):
    sp = (len(clat) - x.shape[0])//2
    return xr.DataArray(
        data = x,
        dims = ["lat","lon"],
        coords = dict(
            lat = (["lat"],clat[sp:-sp]),
            lon  = (["lon"],clon[sp:-sp])
        )
    )
def fromtorch(x,clat,clon):
    x = x.detach().numpy().reshape([x.shape[-2],x.shape[-1]])
    return fromnumpy(x,clat,clon)
