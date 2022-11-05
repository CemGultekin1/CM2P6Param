import os
import sys
import torch
from data.load import get_data
from data.vars import get_var_mask_name
from models.load import load_model
import matplotlib.pyplot as plt
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
from utils.paths import EVALS
from utils.slurm import flushed_print
import numpy as np
from utils.xarray import fromtorchdict, remove_normalization
import xarray as xr

def err_scale_dataset(mean,truef):
    err = np.square(truef - mean)
    sc2 = np.square(truef)
    names = list(err.data_vars)
    for name in names:
        err = err.rename({name : name+'_mse'})
        sc2 = sc2.rename({name : name+'_sc2'})
    return xr.merge([err,sc2])

def expand_depth(evs,depthval):
    print(depthval)
    return evs.expand_dims(dims = dict(depth = depthval),axis=0)
def main():
    args = sys.argv[1:]
    modelid,_,net,_,_,_,_,runargs=load_model(args)
    device = get_device()
    runargs,_ = options(args,key = "run")
    linsupres = runargs.linsupres
    assert runargs.mode == "eval"
    multidatargs = populate_data_options(args,non_static_params=["depth"])
    # multidatargs = multidatargs[:2]
    total_evs = []
    if linsupres:
        total_lsrp_evs = []
    for datargs in multidatargs:
        try:
            test_generator, = get_data(datargs,half_spread = net.spread, torch_flag = False, data_loaders = True,groups = ('test',))
        except GroupNotFoundError:
            print('data not found!')
            continue
        evs = None
        nt_limit = np.inf
        nt = 0
        for fields,forcings,field_masks,forcing_masks,field_coords,forcing_coords in test_generator:
            fields = fromtorchdict(fields,field_coords)
            forcings = fromtorchdict(forcings,forcing_coords)
            field_masks = fromtorchdict(field_masks,field_coords)
            forcing_masks = fromtorchdict(forcing_masks,forcing_coords)
            # fields = mask_dataset(fields,field_masks)
            # forcings = mask_dataset(forcings,forcing_masks)
            
            fields1 = remove_normalization(fields)
            forcings1 = remove_normalization(forcings)
            torch_fields = torch.stack([val for _,val in fields1.data_vars.items],axis = 0)
            torch_fields = torch.stack([torch_fields],axis = 0)
            with torch.set_grad_enabled(False):
                mean,_ =  net.forward(torch_fields.to(device))
                mean = mean.to("cpu")
            if linsupres:
                true_forcing,lsrp_res = separate_linsupres(forcings)
                mean = match(mean,lsrp_res)
                mean = change_scale(mean,denormalize=True)
                true_forcing = change_scale(true_forcing,denormalize=True)
                lsrp_res = change_scale(lsrp_res,denormalize=True)
                lsrp_res = override_names(lsrp_res,true_forcing)
                mean = override_names(mean,true_forcing)
            else:
                true_forcing = forcings
                mean = match(mean,true_forcing)
                mean = change_scale(mean,denormalize=True)
                true_forcing = change_scale(true_forcing,denormalize=True)
            true_forcing = mask(true_forcing,forcing_masks)
            mean = mask(mean,forcing_masks)
            mean = to_xarray(mean,depth,)
            true_forcing = to_xarray(true_forcing,depth,)

            if linsupres:
                lsrp_res = mask(lsrp_res,forcing_masks)
                lsrp_res = to_xarray(lsrp_res,depth)
                lsrp_out = true_forcing - lsrp_res
                mean = mean + lsrp_out
                part_lsrp_evs = err_scale_dataset(lsrp_out,true_forcing)
            part_evs = err_scale_dataset(mean,true_forcing)

            if evs is None:
                evs = part_evs
                if linsupres:
                    lsrp_evs = part_lsrp_evs
            else:
                evs = evs + part_evs
                if linsupres:
                    lsrp_evs = lsrp_evs + part_lsrp_evs
            nt+=1
            if nt>=nt_limit:
                break
        total_evs.append(evs/nt)
        if linsupres:
            total_lsrp_evs.append(lsrp_evs/nt)
        
    evs = xr.merge(total_evs)
    fileame = os.path.join(EVALS,modelid+'.nc')
    evs.to_netcdf(fileame,mode = 'w')
    if linsupres:
        lsrp_evs = xr.merge(total_lsrp_evs)
        fileame = os.path.join(EVALS,'lsrp.nc')
        lsrp_evs.to_netcdf(fileame,mode = 'w')


            

            






if __name__=='__main__':
    main()
