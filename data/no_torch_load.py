import os
from typing import List, Tuple
from data.exceptions import RequestDoesntExist
from data.low_res_dataset import MultiDomainDataset
from data.high_res import  HighResCm2p6, ProjectedHighResCm2p6
from data.paths import get_high_res_data_location, get_high_res_grid_location, get_low_res_data_location
import copy
from data.vars import FIELD_NAMES, FORCING_NAMES, LATITUDE_NAMES, LSRP0_RES_NAMES, LSRP1_RES_NAMES, get_var_mask_name, rename
from data.scalars import load_scalars
import xarray as xr
from data.coords import  DEPTHS, REGIONS, TIMES
from utils.arguments import options
import numpy as np


def load_grid(ds:xr.Dataset,):
    grid_loc = xr.open_dataset(get_high_res_grid_location())
    # import matplotlib.pyplot as plt
    # grid_loc.area_u.plot()
    # plt.savefig('area_u.png')

    passkeys = ['xu_ocean','yu_ocean','xt_ocean','yt_ocean','dxu','dyu','dxt','dyt']#,'area_t',]
    for key in passkeys:
        ds[key] = grid_loc[key]
    return ds

def pass_geo_grid(ds,sigma):
    grid = xr.open_dataset(get_high_res_grid_location())
    lon = grid.xt_ocean.values
    lat = grid.yt_ocean.values
    ilon = np.argsort(((lon + 180 )%360 - 180))
    lon = lon[ilon]
    geolon =grid.geolon_t.values[:,ilon]
    geolat = grid.geolat_t.values[:,ilon]
    geoc = xr.Dataset(
        data_vars = dict(
            geolon = (['lat','lon'],geolon),
            geolat = (['lat','lon'],geolat),
        ),
        coords = dict(
            lat = lat,lon = lon
        )
    )
    if sigma > 1:
        geoc = geoc.coarsen(dim = {'lat':sigma,'lon':sigma},boundary = 'trim').mean().load()
    rlat = ds['lat'].values
    rlon = ds['lon'].values
    # geoc = geoc.sel(lat = slice(rlat[0],rlat[-1]),lon = slice(rlon[0],rlon[-1]))
    geolon = geoc.geolon.values
    geolat = geoc.geolat.values
    def match_shape(g,r,axis):
        df = len(r) - g.shape[axis]
        ldf = df//2
        rdf = df - ldf
        if df>0:
            padding_ = (rdf,ldf)
            padding = [(0,0),(0,0)]
            padding[axis] = padding_
            return np.pad(g,padding,'wrap')
        elif df<0:
            ldf = -ldf
            rdf = -rdf
            return g[ldf:-rdf]
        else:
            return g
    ds = ds.assign_coords(
        dict(
            geolon = (['lat','lon'],(geolon + 180 ) % 360 - 180),
            geolat = (['lat','lon'],geolat)))    
    return ds
        




def load_xr_dataset(args):
    runargs,_ = options(args,'run')
    if runargs.mode == 'data':
        data_address = get_high_res_data_location(args)
    else:
        data_address = get_low_res_data_location(args)
    if not os.path.exists(data_address):
        raise RequestDoesntExist
    ds_zarr= xr.open_zarr(data_address,consolidated=False )
    if runargs.mode == 'data':  
        ds_zarr = load_grid(ds_zarr)
    if runargs.sanity:
        ds_zarr = ds_zarr.isel(time = slice(0,1))
    ds_zarr,scs=  preprocess_dataset(args,ds_zarr)
    return ds_zarr,scs

def get_var_grouping(args)-> Tuple[Tuple[List[str],...],Tuple[List[str],...]]:
    runprms,_=options(args,key = "run")
    fields,forcings = FIELD_NAMES.copy(),FORCING_NAMES.copy()
    lsrp_res = [LSRP0_RES_NAMES.copy(),LSRP1_RES_NAMES.copy()]
    
    if not runprms.temperature and not runprms.mode == 'scalars':
        fields = fields[:2]
        forcings = forcings[:2]
        lsrp_res = [lr[:2] for lr in lsrp_res]
    if runprms.latitude:
        fields.extend(LATITUDE_NAMES)
    varnames = [fields]
    forcingmask_names = []
   
    fieldmasks = [get_var_mask_name(f) for f in fields]
    fieldmask_names = [fieldmasks]

    forcingmasks = [get_var_mask_name(f) for f in forcings]
    lsrpforcingmasks = [[get_var_mask_name(f) for f in lr] for lr in lsrp_res]
    if runprms.mode == 'scalars':
        varnames[0].extend(forcings + lsrp_res[0] + lsrp_res[1])
        forcingmask_names.append(fieldmasks)
        forcingmask_names[0].extend(forcingmasks + lsrpforcingmasks[0] + lsrpforcingmasks[1])
    elif runprms.lsrp>0:
        lsrpi = runprms.lsrp - 1
        if runprms.mode != 'train':
            varnames.append(forcings + lsrp_res[lsrpi])
            forcingmask_names.append(forcingmasks + lsrpforcingmasks[lsrpi])
        else:
            varnames.append(lsrp_res[lsrpi])
            forcingmask_names.append(lsrpforcingmasks[lsrpi])
    else:
        varnames.append(forcings)
        forcingmask_names.append(forcingmasks)
    if runprms.mode == 'view':
        varnames.extend(fieldmask_names)
    varnames.extend(forcingmask_names)
    # print('len(varnames)',len(varnames))
    

    for i in range(len(varnames)):
        varnames[i] = tuple(varnames[i])
    varnames = tuple(varnames)
    return varnames

def dataset_arguments(args,**kwargs_):
    

    prms,_=options(args,key = "data")
    runprms,_=options(args,key = "run")
    
    
    ds_zarr,scalars = load_xr_dataset(args)
    if runprms.mode == 'train':
        boundaries = REGIONS[prms.domain]
    else:
        boundaries = REGIONS['global']
        
    kwargs = ['lsrp','parts','latitude','temperature','section']
    if runprms.mode == 'eval':
        kwargs.pop(1)
    kwargs = {key:runprms.__dict__[key] for key in kwargs}
    kwargs['boundaries'] = boundaries
    kwargs['scalars'] = scalars
    kwargs['coarse_grain_needed'] = runprms.mode == "data"

    for key,val in kwargs_.items():
        kwargs[key] = val
    def isarray(x):
        return isinstance(x,list) or isinstance(x,tuple)
    for key,var in kwargs.items():
        if isarray(var):
            if isarray(var[0]):
                for i in range(len(var)):
                    var[i] = tuple(var[i])
                kwargs[key] = list(var)
            else:
                kwargs[key] = tuple(var)
    kwargs['var_grouping'] = get_var_grouping(args)
    args = (ds_zarr,prms.sigma)
    return args,kwargs



def load_lowres_dataset(args,**kwargs)->List[MultiDomainDataset]:
    _args,_kwargs = dataset_arguments(args,**kwargs)
    ds = MultiDomainDataset(*_args, **_kwargs)
    dsets = populate_dataset(ds,**kwargs)
    return dsets

def load_highres_dataset(args,**kwargs)->HighResCm2p6:
    _args,_kwargs = dataset_arguments(args,**kwargs)
    ds = HighResCm2p6(*_args, **_kwargs)
    return (ds,)



def get_data(args,**kwargs):
    ns,_ = options(args,key = "run")
    if ns.mode != "data":
        dsets = load_lowres_dataset(args,**kwargs)
    else:
        dsets = load_highres_dataset(args,**kwargs)

    return dsets


def populate_dataset(dataset:MultiDomainDataset,groups = ("train","validation"),**kwargs):
    datasets = []
    for group in groups:
        t0,t1 = TIMES[group]
        datasets.append(dataset.set_time_constraint(t0,t1))
    return tuple(datasets)
def get_time_values(deep):
    if deep:
        return load_xr_dataset('--mode train --depth 5'.split())[0].time.values
    return load_xr_dataset('--mode train --depth 0'.split())[0].time.values

def preprocess_dataset(args,ds:xr.Dataset):
    prms,_ = options(args,key = "run")
    ds = rename(ds)
    coord_names = list(ds.coords.keys())

    def add_co2(ds,prms):
        if prms.co2:
            ds['co2'] = [0.01]
        else:
            ds['co2'] = [0.]
        ds = ds.isel(co2 = 0)
        return ds
    ds = add_co2(ds,prms)
    if prms.depth > 1e-3:
        if 'depth' not in coord_names:
            raise RequestDoesntExist
        if prms.mode == 'data':
            depthvals_=ds.coords['depth'].values
            inds = [np.argmin(np.abs(depthvals_ - d )) for d in DEPTHS if d>1e-3]
            ds = ds.isel(depth = inds)
        else:
            depthvals_=ds.coords['depth'].values
            ind = np.argmin(np.abs(depthvals_ - prms.depth ))
            ds = ds.isel(depth = ind)
            if np.abs(ds.depth.values-prms.depth)>1:
                print(f'requested depth {prms.depth},\t existing depth = {ds.depth.values}')
                raise RequestDoesntExist
    else:
        ds['depth'] = [0]
        if prms.mode != 'data':
            ds = ds.isel(depth = 0)
    if prms.mode in ['train','eval','view']:
        depthval = ds.depth.values
        trd = ds.tr_depth.values
        tr_ind = np.argmin(np.abs(depthval - trd))
        if np.abs(trd[tr_ind] - depthval)>1:
            raise RequestDoesntExist
        ds = ds.isel(tr_depth = tr_ind)
    scs = load_scalars(args)
    if prms.mode != 'scalars' and scs is not None and prms.mode != 'data' :
        depthval = ds.depth.values
        trd = scs.tr_depth.values
        tr_ind = np.argmin(np.abs(depthval - trd))
        if np.abs(trd[tr_ind] - depthval)>1:
            raise RequestDoesntExist
        scs = scs.isel(tr_depth = tr_ind)
    return ds,scs

def physical_domains(domain:str,):
    partition={}
    parts = ['train','validation','test']
    for part in parts:
        partition[part]=copy.deepcopy(REGIONS[domain])
        for key,val in TIMES[part].items():
            partition[part][key] = val
    return partition
