import os
from typing import List, Tuple
from data.exceptions import RequestDoesntExist
from data.gcm_dataset import MultiDomainDataset
from data.paths import get_high_res_data_location, get_low_res_data_location
import copy
from data.vars import FIELD_NAMES, FORCING_NAMES, LATITUDE_NAMES, LSRP_RES_NAMES, get_var_mask_name, rename
from utils.paths import SCALARS_JSON
import xarray as xr
from data.coords import  REGIONS, TIMES
from utils.arguments import options
import numpy as np
import json
import torch

depthvals=np.array([0.,5.03355 , 55.853249,  110.096153, 181.312454,  330.007751,1497.56189 ])
sigmavals=np.array([4,8,12,16])
domainvals = ["four_regions","global"]
normalizationvals =  ["standard","absolute"]


def load_scalars(args,):
    run,_ = options(args,key = "run")
    if run.mode == "data" or run.mode == "scalars":
        return None
    _,scalarid = options(args,key = "scalar")
    file = SCALARS_JSON
    if os.path.exists(file):
        with open(file) as f:
            scalars = json.load(f)
        if scalarid in scalars:
            return scalars[scalarid]

    print('scalars are not found!')
    return None

def load_xr_dataset(args):
    runargs,_ = options(args,'run')
    if runargs.mode == 'data':
        data_address = get_high_res_data_location(args)
    else:
        data_address = get_low_res_data_location(args)
    print('data_address',data_address)
    ds_zarr= xr.open_zarr(data_address)
    ds_zarr =  preprocess_dataset(args,ds_zarr)
    return ds_zarr

def get_var_grouping(args)-> Tuple[Tuple[List[str],...],Tuple[List[str],...]]:
    runprms,_=options(args,key = "run")
    fields,forcings = FIELD_NAMES.copy(),FORCING_NAMES.copy()
    lsrp_res_forcings = LSRP_RES_NAMES.copy()
    if not runprms.temperature:
        fields = fields[:2]
        forcings = forcings[:2]
        lsrp_res_forcings = lsrp_res_forcings[:2]
    if runprms.latitude:
        fields.extend(LATITUDE_NAMES)
    varnames = [fields]
    masknames = []
    forcingmasks = [get_var_mask_name(f) for f in forcings]
    lsrpresforcingmasks = [get_var_mask_name(f) for f in lsrp_res_forcings]
    if runprms.linsupres:
        if runprms.mode != 'train':
            varnames.append(forcings)
            masknames.append(forcingmasks)
        varnames.append(lsrp_res_forcings)
        masknames.append(lsrpresforcingmasks)
    else:
        varnames.append(forcings)
        masknames.append(forcingmasks)
    varnames.extend(masknames)

    for i in range(len(varnames)):
        varnames[i] = tuple(varnames[i])
    varnames = tuple(varnames)
    return varnames

def dataset_arguments(args,**kwargs_):
    ds_zarr = load_xr_dataset(args)
    prms,_=options(args,key = "data")
    runprms,_=options(args,key = "run")

    scalars_dict = load_scalars(args)
    boundaries = REGIONS[prms.domain]
    kwargs = ['linsupres','parts','latitude','normalization','lsrp_span','temperature']
    kwargs = {key:runprms.__dict__[key] for key in kwargs}
    kwargs['boundaries'] = boundaries
    kwargs['scalars_dict'] = scalars_dict
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

def load_dataset(args,**kwargs)->MultiDomainDataset:
    _args,_kwargs = dataset_arguments(args,**kwargs)
    ds = MultiDomainDataset(*_args, **_kwargs)
    dsets = populate_dataset(ds,**kwargs)
    return dsets


class TorchDatasetWrap(torch.utils.data.Dataset):
    def __init__(self,mdm):
        self.mdm = mdm
    def __len__(self,):
        return self.mdm.__len__()
    def __getitem__(self,i):
        return self.mdm[i]

def get_data(args,torch_flag = False,data_loaders = True,**kwargs):
    dsets = load_dataset(args,torch_flag = torch_flag,**kwargs)

    if data_loaders:
        ns,_ = options(args,key = "run")
        params={'batch_size':ns.minibatch,\
            'shuffle':ns.mode == "train",\
            'num_workers':ns.num_workers,\
            'prefetch_factor':ns.prefetch_factor,\
            'persistent_workers':ns.persistent_workers,}
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
    prms,_ = options(args,key = "data")
    ds = rename(ds)
    coord_names = list(ds.coords.keys())

    if prms.depth > 0:
        if 'depth' not in coord_names:
            raise RequestDoesntExist
        depthvals_=ds.coords['depth'].values
        ind = np.argmin(np.abs(depthvals_ - prms.depth ))
        ds = ds.isel(depth = ind)
        if int(ds.depth.values) != int(prms.depth):
            raise RequestDoesntExist
    else:
        ds = ds.expand_dims(dim = {"depth" : [0]},axis=0).isel(depth = 0)
    return ds

def physical_domains(domain:str,):
    partition={}
    parts = ['train','validation','test']
    for part in parts:
        partition[part]=copy.deepcopy(REGIONS[domain])
        for key,val in TIMES[part].items():
            partition[part][key] = val
    return partition
