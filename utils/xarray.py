from typing import Callable
import xarray as xr
import torch
import numpy as np
import torch.nn as nn
def no_nan_input_mask(u,span,codition:Callable = lambda x:  np.isnan(x))->xr.DataArray:
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
    # pool = nn.MaxPool2d(2*sp + 1,stride = 1)
    pool = nn.Conv2d(1,1,2*sp+1,bias = False)
    pool.weight.data = torch.ones(1,1,2*sp+1,2*sp+1).type(torch.float32)/((2*sp+1)**2)
    with torch.no_grad():
        poolmask = pool(torchmask).numpy().reshape(*shpp)
    mask = np.ones(shp,dtype = float)
    mask[sp:-sp,sp:-sp] = poolmask
    mask = xr.DataArray(
        data = mask,
        dims = ["lat","lon"],
        coords = dict(
            lat = (["lat"],u.lat.values),
            lon = (["lon"],u.lon.values)
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
