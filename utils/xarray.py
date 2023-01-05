from typing import Callable
from data.vars import get_var_mask_name
import xarray as xr
import torch
import numpy as np
import torch.nn as nn
from scipy.ndimage import gaussian_filter

def skipna_mean(ds,dim):
    _nonancount= xr.where(np.isnan(ds) + (np.abs(ds) == np.inf  ),0,1)
    _values = xr.where(np.isnan(ds) + (np.abs(ds) == np.inf ),0,ds)
    mean_val =  _values.sum(dim = dim)/_nonancount.sum(dim = dim)
    assert np.all(1 - np.isnan(mean_val))
    return mean_val

def land_fill(u_:xr.DataArray,factor,ntimes,zero_tendency = False):
    u = u_.copy()
    for _ in range(ntimes):
        u0 = u.fillna(0).values
        if zero_tendency:
            wetmask = xr.where(np.isnan(u),1,1).values
        else:
            wetmask = xr.where(np.isnan(u),0,1).values
        
        u0bar = gaussian_filter(u0*wetmask,sigma = factor,mode= 'constant',cval = np.nan)
        wetbar = gaussian_filter(wetmask.astype(float),sigma = factor,mode= 'constant',cval = np.nan)
        u0bar = u0bar/wetbar
        u0bar = xr.DataArray(
            data = u0bar,
            dims = u.dims,
            coords = u.coords
        )
        u = xr.where(np.isnan(u),u0bar,u)
    return u
def plot_ds(ds,imname,ncols = 3,dims = ['lat','lon'],\
    cmap = 'seismic',\
    lognorm = False,scale_grouping = []):
    kwargs = dict(dims = dims,cmap = cmap)
    if isinstance(ds,list):
        for i,ds_ in enumerate(ds):
            imname_ = imname.replace('.png',f'-{i}.png')
            plot_ds(ds_,imname_,ncols = ncols,**kwargs)
        return

    if isinstance(ds,dict):
        for name,var in ds.items():
            var.name = name
        plot_ds(xr.merge(list(ds.values())),imname,ncols = ncols,**kwargs)
        return
    import matplotlib.pyplot as plt
    import matplotlib
    import itertools
    import matplotlib.colors as mcolors
    
    
    excdims = []
    for key in ds.data_vars.keys():
        u = ds[key]
        dim = list(u.dims)
        excdims.extend(dim)
    excdims = np.unique(excdims).tolist()
    for d in dims:
        flag = -1
        for ii,excd in enumerate(excdims):
            if d in excd:
                flag = ii
                break        
        if flag == -1:
            raise Exception
        excdims.pop(flag)

    flat_vars = {}
    for key in ds.data_vars.keys():
        u = ds[key]
        eds = [d for d in u.dims if d in excdims if len(ds.coords[d])>1]
        base_sel = {d : 0 for d in u.dims if d in excdims if len(ds.coords[d])==1}
        neds = [len(ds.coords[d]) for d in eds]
        inds = [range(nd) for nd in neds]
        for multi_index in itertools.product(*inds):
            secseldict = {ed:mi for ed,mi in zip(eds,multi_index)}
            seldict = dict(base_sel,**secseldict)
            keyname = key + '_'.join([f"{sk}_{si}" for sk,si in secseldict.items()])
            flat_vars[keyname] = u.isel(**seldict)
    vars = list(flat_vars.keys())
    nrows = int(np.ceil(len(vars)/ncols))
    fig,axs = plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*5))
    print('nrows,ncols\t',nrows,ncols)

    def listify(cmap):
        if not isinstance(cmap ,list):
            cmap = [cmap]*nrows
        cmap = cmap + [cmap[-1]]*(nrows - len(cmap))
        return cmap

    cmap = listify(cmap)
    lognorm = listify(lognorm)


    def find_min_max(name):
        if not scale_grouping:
            return dict()
        found_flag = False
        for psc,grouping in scale_grouping:
            if name in grouping:
                found_flag = True
                break
        if not found_flag:
            return dict()
        vmax = -np.inf
        for var in grouping:
            vmax_ = np.amax(np.abs(flat_vars[var].fillna(0).values))
            vmax = np.maximum(vmax,vmax_)
            if psc:
                break
        return dict(vmin = -vmax,vmax = vmax)




    for z,(i,j) in enumerate(itertools.product(range(nrows),range(ncols))):
        if nrows == 1 and ncols == 1:
            ax = axs
        elif nrows == 1:
            ax = axs[j]
        elif ncols == 1:
            ax = axs[i]
        else:
            ax = axs[i,j]
        if z >= len(vars):
            continue
        
        u = flat_vars[vars[z]]
        
        cmap_ = matplotlib.cm.get_cmap(cmap[i])
        cmap_.set_bad('black',.4)

        if lognorm[i]:
            norm = mcolors.LogNorm()
        else:
            norm = mcolors.Normalize()
        u.plot(ax = ax,cmap = cmap_,norm = norm,**find_min_max(vars[z]))
        ax.set_title(vars[z])
        ax.set_ylabel('')
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

def make_dimensional(x:xr.Dataset,coordname,coordval):
    dv,c = tonumpydict(x)
    ndv = {}
    for name,(dims,vals) in dv.items():
        ndv[name] = ([coordname],vals.reshape([-1]))
    c[coordname] = np.array([coordval])
    return xr.Dataset(data_vars = ndv,coords = c)

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
def fromtorchdict2tensor(data_vars,contained = '',**kwargs):
    vecs = []
    for key in data_vars:
        if '_scale' in key:
            continue
        if contained not in key:
            continue
        _,vec = data_vars[key]
        vecs.append(vec)
    for i in range(len(vecs)):
        vec = vecs[i]
        j = 4 - len(vec.shape)
        vecs[i] = vec.reshape([1]*j + list(vec.shape))
    return torch.cat(vecs,dim=1)

def fromtensor2dict(tts,data_vars0,contained = '',**kwargs):
    data_vars = {}
    i= 0 
    for key in data_vars0:
        dims,vec = data_vars0[key]
        if '_scale' in key:
            data_vars[key] = dims,vec
        elif contained in key:
            data_vars[key] = dims, tts[0,i]
            i+=1
    return data_vars

def fromtensor(tts,data_vars0,coords,masks,denormalize = True,fillvalue = np.nan,**kwargs):
    data_vars = fromtensor2dict(tts,data_vars0,**kwargs)
    return fromtorchdict(data_vars,coords,masks,normalize = False,denormalize=denormalize, fillvalue=fillvalue,**kwargs)

def fromtorchdict(data_vars,coords,masks,normalize = False,denormalize = False,fillvalue = np.nan,**kwargs):
    ds = fromtorchdict2dataset(data_vars,coords)
    dsmasks = fromtorchdict2dataset(masks,coords)
    ds = mask_dataset(ds,dsmasks,fillvalue = fillvalue)
    if normalize:
        ds = normalize_dataset(ds,denormalize=False,**kwargs)
    elif denormalize:
        ds = normalize_dataset(ds,denormalize=True,**kwargs)
    ds = remove_normalization(ds)
    return drop_unused_coords(ds,**kwargs)
def drop_unused_coords(ds,expand_dims = {},**kwargs):
    cns = list(ds.coords.keys())
    dims = []
    for key in ds.data_vars.keys():
        dims.extend(list(ds[key].dims))
    dims = np.unique(dims)
    dropcns = [c for c in cns if c not in dims and c not in expand_dims]
    for dcn in dropcns:
        ds = ds.drop(dcn)
    keys = list(expand_dims.keys())
    for i in range(len(keys)):
        key = keys[-i-1]
        ds = ds.expand_dims(dim = {key:expand_dims[key]},axis=0)
    return ds
def fromtorchdict2dataset(data_vars,coords):
    for key,(dims,vals) in data_vars.items():
        if isinstance(vals,torch.Tensor):
            vals = vals.numpy()
        data_vars[key] = (dims,vals)
    for key,val in coords.items():
        if isinstance(val,torch.Tensor):
            coords[key] = val.numpy()
    ds = xr.Dataset(data_vars = data_vars,coords = coords)
    return ds

def normalize_dataset(ds,denormalize = False,drop_normalization = False,**kwargs):
    # print(list(ds.data_vars.keys()))
    for key in ds.data_vars.keys():
        if '_scale' in key : # if 'mean' in key or 'std' in key:
            continue
        # mkey = f"{key}_mean"
        # skey = f"{key}_std"
        mkey = f"{key}_scale"
        sc = ds[mkey].values
        
        # a,b = ds[mkey].values,ds[skey].values

        dims1 = ds[key].dims
        dims0 = ds[mkey].dims
        shp = []
        for d in dims1:
            if d in dims0:
                shp.append(len(ds[d]))
            else:
                shp.append(1)
        # a,b = a.reshape(shp),b.reshape(shp)
        if denormalize:
            ds[key] = ds[key] * sc #b + a
        else:
            ds[key] = ds[key]/sc #(ds[key] - a)/b
        if drop_normalization:
            ds = ds.drop(mkey)
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
        if '_scale' not in key:
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
    coords = dict()
    for name,var in kwargs.items():
        data_vars[name] = (var.dims,var.data)
        coords_ = {key:coo for key,coo in var.coords.items() if key in var.dims}
        coords = dict(coords,**coords_)
    return xr.Dataset(data_vars = data_vars,coords = coords)

def unbind(ds):
    d = dict()
    for key in ds.data_vars.keys():
        d[key] = ds[key]
    return d



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
