import sys
import torch
from data.load import get_data
from data.vars import get_var_mask_name
from models.load import load_model
import matplotlib.pyplot as plt
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
from utils.xarray import numpydict2dataset
import numpy as np
import xarray as xr

def change_scale(*dicts,normalize = False,denormalize = False):
    dicts = list(dicts)
    groups = [list(d.keys()) for d in dicts]
    cdict = dicts[0]
    for d in dicts[1:]:
        cdict = dict(cdict,**d)
    valuesdict = {}
    for key,val in cdict.items():
        f = val['val']
        n = val['normalization']
        valuesdict[key] = {}
        if normalize:
            valuesdict[key]['val'] = (f - n[:,0].reshape([-1,1,1]))/n[:,1].reshape([-1,1,1])
        elif denormalize:
            valuesdict[key]['val'] = f*n[:,1].reshape([-1,1,1]) + n[:,0].reshape([-1,1,1])
    valuesdict = pass_other_keys(valuesdict,cdict)
    newdicts = []
    for g in groups:
        newdicts.append({key:valuesdict[key] for key in g})
    return tuple(newdicts)

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
        n = forcings[key]['normalization']
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
def to_xarray(torchdict,times):
    data_vars = {
        key : (["time","lat","lon"] ,torchdict[key]['val'].numpy()) for key in torchdict
    }
    for key in torchdict:
        lat = torchdict[key]["lat"][0,:].numpy()
        lon = torchdict[key]["lon"][0,:].numpy()
        break
    coords = dict(time = (["time"],times),lat = (["lat"],lat),lon = (["lon"],lon))
    return xr.Dataset(data_vars = data_vars,coords = coords)
def apply_keywise(*dicts,fun = lambda x: x):
    newdict = {}
    dicts = list(dicts)
    for key in dicts[0]:
        newdict[key] = {}
        vecs = [dicts[i][key]['val'] for i in range(len(dicts))]
        newdict[key]['val'] = fun(vecs)
    return pass_other_keys(newdict,dicts[0],exceptions = ['val'])

def err_scale_dataset(mean,truef):
    err = truef - mean
    sc2 = np.square(truef)
    names = list(err.data_vars)
    for name in names:
        err = err.rename({name : name+'_err'})
        sc2 = sc2.rename({name : name+'_sc2'})
    
    return xr.merge([err,sc2])
def main():
    args = sys.argv[1:]
    # modelid,_,net,_,_,_,logs,runargs=load_model(args)
    # device = get_device()
    runargs,_ = options(args,key = "run")
    assert runargs.mode == "eval"
    multidatargs = populate_data_options(args,)
    for datargs in multidatargs:
        test_generator, = get_data(datargs,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('test',))
        dataprms, _ = options(datargs,key = "data")
        linsupres = dataprms.linsupres
        for outputs in test_generator:
            fields,forcings,forcing_masks = outputs
            torch_fields, = torch_stack(fields)
            with torch.set_grad_enabled(False):
                # mean,_ =  net.forward(torch_fields.to(device))
                # mean = mean.to("cpu")
                mean = torch.randn(1,3,607, 898,dtype = torch.float32)
            if linsupres:
                true_forcing,lsrp_res = separate_linsupres(forcings)
                mean = match(mean,lsrp_res)
                mean,true_forcing,lsrp_res, = change_scale(mean,true_forcing,lsrp_res, denormalize=True)
                lsrp_res = override_names(lsrp_res,true_forcing)
                mean = override_names(mean,true_forcing)
            else:
                true_forcing = forcings
                mean = match(mean,true_forcing)
                mean,true_forcing, = change_scale(mean,true_forcing,denormalize=True)
            times = np.arange(1)
            true_forcing = mask(true_forcing,forcing_masks)
            mean = mask(mean,forcing_masks)
            mean = to_xarray(mean,times)
            true_forcing = to_xarray(true_forcing,times)
            if linsupres:
                lsrp_res = mask(lsrp_res,forcing_masks)
                lsrp_res = to_xarray(lsrp_res,times)
                lsrp_out = true_forcing - lsrp_res
                mean = mean + lsrp_out
            evs = err_scale_dataset(mean,true_forcing)

            print(evs)
            return False
            # if linsupres:
            #     lsrp_output = apply_keywise(true_forcing,lsrp_res,fun = lambda a,b : a-b)
            #     mean = apply_keywise(mean,lsrp_output,fun = lambda a,b : a+b)
            

            






if __name__=='__main__':
    main()
