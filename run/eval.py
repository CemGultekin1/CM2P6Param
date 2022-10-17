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
import numpy as np
import xarray as xr

def change_scale(d0,normalize = False,denormalize = False):
    for key,val in d0.items():
        f = val['val']
        n = val['normalization']
        if normalize:
            d0[key]['val'] = (f - n[:,0].reshape([-1,1,1]))/n[:,1].reshape([-1,1,1])
        elif denormalize:
            d0[key]['val'] = f*n[:,1].reshape([-1,1,1]) + n[:,0].reshape([-1,1,1])
    return d0

def torch_stack(*dicts):
    dicts = list(dicts)
    groups = [list(d.keys()) for d in dicts]
    cdict = dicts[0]
    for d in dicts[1:]:
        cdict = dict(cdict,**d)
    newdicts = []
    for g in groups:
        newdicts.append(torch.stack([cdict[key]['val'] for key in g],dim = 1))
    return tuple(newdicts)

def mask(outputs,masks):
    for key in outputs:
        m = masks[get_var_mask_name(key)]['val']
        vec = outputs[key]['val']
        mask = m<0.5
        vec[mask] = np.nan
        outputs[key]['val'] = vec
    return outputs

def match(outputs,forcings,):
    outputs = torch.unbind(outputs, dim = 1)
    keys = list(forcings.keys())
    outputdict = {}
    for i,(out,key) in enumerate(zip(outputs,keys)):
        outputdict[key] = {}
        outputdict[key]['val'] = out
    outputdict = pass_other_keys(outputdict,forcings)
    return outputdict

def concat_datasets(x,y):
    for key in y:
        for key_ in y[key]:
            v0 = x[key][key_]
            v1 = y[key][key_]
            x[key][key_] = torch.cat((v0,v1),dim = 0)
    return x
def separate_linsupres(forcings):
    keys = list(forcings.keys())
    nk = len(keys)//2
    true_forcings = {key:forcings[key] for key in keys[:nk]}
    lsrp_res_forcings = {key:forcings[key] for key in keys[nk:]}
    return true_forcings,lsrp_res_forcings

def override_names(d1,d2):
    newdict = {}
    keys1 = list(d1.keys())
    keys2 = list(d2.keys())
    for key1,key2 in zip(keys1,keys2):
        newdict[key2] = d1[key1]
    return newdict
def pass_other_keys(d1,d2,exceptions = ['val']):
    for key in d2:
        for k in d2[key]:
            if k in exceptions:
                continue
            d1[key][k] = d2[key][k]
    return d1

def to_xarray(torchdict,depth):
    data_vars = {
        key : (["depth","lat","lon"] ,torchdict[key]['val'].numpy()) for key in torchdict
    }
    for key in torchdict:
        lat = torchdict[key]["lat"][0,:].numpy()
        lon = torchdict[key]["lon"][0,:].numpy()
        break
    coords = dict(lat = (["lat"],lat),lon = (["lon"],lon),depth = (["depth"],depth))
    return xr.Dataset(data_vars = data_vars,coords = coords)

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
            test_generator = None
        if test_generator is None:
            continue
        evs = None
        nt_limit = np.inf
        nt = 0
        for fields,forcings,forcing_masks,info in test_generator:
            depth = info['depth'].numpy().reshape([-1])
            if nt == 0:
                print(depth[0])

            torch_fields, = torch_stack(fields)
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
