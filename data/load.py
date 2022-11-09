import os
from typing import List, Tuple
from data.exceptions import RequestDoesntExist
from data.low_res_dataset import MultiDomainDataset
from data.generate import  ProjectedHighResCm2p6
from data.paths import get_high_res_data_location, get_high_res_grid_location, get_low_res_data_location
import copy
from data.vars import FIELD_NAMES, FORCING_NAMES, LATITUDE_NAMES, LSRP0_RES_NAMES, LSRP1_RES_NAMES, get_var_mask_name, rename
from data.scalars import load_scalars
import xarray as xr
from data.coords import  DEPTHS, REGIONS, TIMES
from utils.arguments import options
import numpy as np
import torch


def load_grid(ds:xr.Dataset,):
    grid_loc = xr.open_dataset(get_high_res_grid_location())
    # import matplotlib.pyplot as plt
    # grid_loc.area_u.plot()
    # plt.savefig('area_u.png')

    passkeys = ['xu_ocean','yu_ocean','xt_ocean','yt_ocean','dxu','dyu','dxt','dyt']#,'area_t',]
    for key in passkeys:
        ds[key] = grid_loc[key]
    return ds


def load_xr_dataset(args):
    runargs,_ = options(args,'run')
    if runargs.mode == 'data':
        data_address = get_high_res_data_location(args)
    else:
        data_address = get_low_res_data_location(args)
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
            # varnames.append(fieldmasks)
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
    
    boundaries = REGIONS[prms.domain]
    kwargs = ['lsrp','parts','latitude','lsrp_span','temperature','section']
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

class Dataset(torch.utils.data.Dataset):
    mdm:MultiDomainDataset
    def __init__(self,mdm:MultiDomainDataset):
        self.mdm = mdm
    def __len__(self,):
        return len(self.mdm)
    def __getitem__(self,i):
        outs =  self.mdm[i]
        return self.mdm.outs2numpydict(outs)

def load_lowres_dataset(args,**kwargs)->List[MultiDomainDataset]:
    _args,_kwargs = dataset_arguments(args,**kwargs)
    ds = MultiDomainDataset(*_args, **_kwargs)
    dsets = populate_dataset(ds,**kwargs)
    return dsets

def load_highres_dataset(args,**kwargs)->ProjectedHighResCm2p6:
    _args,_kwargs = dataset_arguments(args,**kwargs)
    ds = ProjectedHighResCm2p6(*_args, **_kwargs)
    return (ds,)

class TorchDatasetWrap(torch.utils.data.Dataset):
    def __init__(self,mdm):
        self.mdm = mdm
    def __len__(self,):
        return self.mdm.__len__()
    def __getitem__(self,i):
        return self.mdm[i]

def get_data(args,torch_flag = False,data_loaders = True,**kwargs):
    ns,_ = options(args,key = "run")
    if ns.mode != "data":
        dsets = load_lowres_dataset(args,torch_flag = torch_flag,**kwargs)
    else:
        dsets = load_highres_dataset(args,torch_flag = torch_flag,**kwargs)

    if data_loaders:
        minibatch = ns.minibatch
        if ns.mode != "train":
            minibatch = None
        params={'batch_size':minibatch,\
            'shuffle':ns.mode == "train" or ns.mode == "view",\
            'num_workers':ns.num_workers,\
            'prefetch_factor':ns.prefetch_factor,\
            'persistent_workers':ns.persistent_workers,}
            # 'pin_memory': True,}
        torchdsets = (TorchDatasetWrap(dset_) for dset_ in dsets)
        return (torch.utils.data.DataLoader(tset_, **params) for tset_ in torchdsets)
    else:
        return dsets


def populate_dataset(dataset:MultiDomainDataset,groups = ("train","validation"),**kwargs):
    datasets = []
    for group in groups:
        t0,t1 = TIMES[group]
        datasets.append(dataset.time_slice(t0,t1))
    return tuple(datasets)


def preprocess_dataset(args,ds:xr.Dataset):
    prms,_ = options(args,key = "run")
    ds = rename(ds)
    coord_names = list(ds.coords.keys())
    
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
            if int(ds.depth.values) != int(prms.depth):
                raise RequestDoesntExist
    else:
        ds['depth'] = [0]
        if prms.mode != 'data':
            ds = ds.isel(depth = 0)
    if prms.mode == 'train' or prms.mode == 'eval':
        depthval = ds.depth.values
        trd = ds.tr_depth.values
        tr_ind = np.argmin(np.abs(depthval - trd))
        if np.abs(trd[tr_ind] - depthval)>1:
            raise RequestDoesntExist
        ds = ds.isel(tr_depth = tr_ind)
    scs = load_scalars(args)
    if prms.mode != 'scalars' and scs is not None:
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
