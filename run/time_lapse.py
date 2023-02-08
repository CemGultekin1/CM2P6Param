import itertools
import os
import sys
from cheng_field_test import load_cheng_model_first_version
from data.exceptions import RequestDoesntExist
from plots.metrics import metrics_dataset, moments_dataset
from run.train import Timer
import torch
from data.load import get_data
from data.vars import get_var_mask_name
from models.load import load_model, load_old_model
import matplotlib.pyplot as plt
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
from utils.paths import TIME_LAPSE
from utils.slurm import flushed_print
import numpy as np
from utils.xarray import concat, fromtensor, fromtorchdict, fromtorchdict2tensor, plot_ds
import xarray as xr

# def change_scale(d0,normalize = False,denormalize = False):
#     for key,val in d0.items():
#         f = val['val']
#         n = val['normalization']
#         if normalize:
#             d0[key]['val'] = (f - n[:,0].reshape([-1,1,1]))/n[:,1].reshape([-1,1,1])
#         elif denormalize:
#             d0[key]['val'] = f*n[:,1].reshape([-1,1,1]) + n[:,0].reshape([-1,1,1])
#     return d0

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
    return evs.expand_dims(dims = dict(depth = depthval),axis=0)
def lsrp_pred(respred,tr):
    keys= list(respred.data_vars.keys())
    data_vars = {}
    coords = {key:val for key,val in tr.coords.items()}
    for key in  keys:
        trkey = key.replace('_res','')
        trval = tr[trkey] - tr[key] # true - (true - lsrp) = lsrp
        data_vars[trkey] = (trval.dims,trval.values)
        respred[key] = trval + respred[key]
        respred = respred.rename({key:trkey})
        tr = tr.drop(key)
    lsrp = xr.Dataset(data_vars =data_vars,coords = coords)
    return (respred,lsrp),tr
def update_stats(stats,prd,tr,key):
    stats_ = moments_dataset(prd,tr)
    if key not in stats:
        stats[key] = stats_
    else:
        stats[key] = stats[key] + stats_
    return stats
def get_lsrp_modelid(args):
    runargs,_ = options(args,key = "model")
    lsrp_flag = runargs.lsrp > 0 and runargs.temperature
    if not lsrp_flag:
        return False, None
    keys = ['model','sigma']
    vals = [runargs.__getattribute__(key) for key in keys]
    lsrpid = runargs.lsrp - 1
    vals[0] = f'lsrp:{lsrpid}'
    line =' '.join([f'--{k} {v}' for k,v in zip(keys,vals)])
    _,lsrpid = options(line.split(),key = "model")
    return True, lsrpid

class CoordinateLocalizer:
    def get_local_ids(self,coord,field_coords):
        cdict = dict(lat = coord[0],lon = coord[1])
        ldict = dict(lat = None,lon = None)

        for key,val in cdict.items():
            i = np.argmin(np.abs(field_coords[key].numpy() - val))
            ldict[key] = i
        return ldict
    
    def get_localized(self,coord,spread,field_coords,*args):
        ldict1 = self.get_local_ids(coord,field_coords)
        if spread > 0:
            latsl,lonsl = [slice(ldict1[key] - spread,ldict1[key] + spread + 1) for key in 'lat lon'.split()]
        else:
            latsl,lonsl = [slice(ldict1[key] ,ldict1[key] + 1) for key in 'lat lon'.split()]
        
        newargs = []
        for arg in args:
            newarg = dict()
            for key,data_var in arg.items():
                dims,vals = data_var
                if len(vals.shape) == 2:
                    newarg[key] = (dims,vals[latsl,lonsl])
                else:
                    newarg[key] = (dims,vals)
            newargs.append(newarg)

        newcoords = dict()
        dslice = dict(lat = latsl,lon = lonsl)
        for key,val in field_coords.items():
            if key not in 'lat lon'.split():
                newcoords[key] = val
            else:
                newcoords[key] = val[dslice[key]]
        newargs = [newcoords] + newargs
        return tuple(newargs)


def main():
    modelid,_,args = load_old_model(int(sys.argv[1]))
    modelid,net = load_cheng_model_first_version()

    
    args.extend('--mode eval --num_workers 1'.split())
    runargs,_ = options(args,key = "run")
    
    device = get_device()
    
    assert runargs.mode == "eval"
    multidatargs = populate_data_options(args,non_static_params=['depth','co2'],domain = 'global')
    stats = None
    coords = [(30,-60),(-20,-104)]
    localizer = CoordinateLocalizer()
    if net is dict:
        spread = net['mean'].spread
    else:
        spread = net.spread
        
    for datargs in multidatargs:
        try:
            test_generator, = get_data(datargs,half_spread = spread, torch_flag = False, data_loaders = True,groups = ('train',))
        except RequestDoesntExist:
            print('data not found!')
            test_generator = None
        if test_generator is None:
            continue
        nt = 0
        for fields,forcings,forcing_mask,field_coords,forcing_coords in test_generator:
            for cid,coord in  enumerate(coords):
                _,loc_fields, = localizer.get_localized(coord, spread,field_coords,fields, )
                loc_forcing_coords,loc_forcings,loc_forcing_mask, = localizer.get_localized(coord, 0,forcing_coords,forcings, forcing_mask)
  
                fields_tensor = fromtorchdict2tensor(loc_fields).type(torch.float32)
                depth = forcing_coords['depth'].item()
                co2 = forcing_coords['co2'].item()
                time = forcing_coords['time'].item()
                kwargs = dict(contained = '', \
                    expand_dims = {'co2':[co2],'depth':[depth],'time' : [time],'coord_id' :[cid]},\
                    drop_normalization = True,
                    )


                with torch.set_grad_enabled(False):
                    if net is dict:
                        mean,_ =  net['mean'].forward(fields_tensor.to(device))
                        _,var =  net['var'].forward(fields_tensor.to(device))
                        mean = mean.to("cpu")
                        var = var.to("cpu")
                        std = torch.sqrt(var)
                    else:
                        mean,std = net.forward(fields_tensor.to(device))
                        std = torch.sqrt(1/std)
                


                predicted_forcings = fromtensor(mean,loc_forcings,loc_forcing_coords, loc_forcing_mask,denormalize = True,**kwargs)
                predicted_std = fromtensor(std,loc_forcings,loc_forcing_coords, loc_forcing_mask,denormalize = True,**kwargs)
                true_forcings = fromtorchdict(loc_forcings,loc_forcing_coords,loc_forcing_mask,denormalize = True,**kwargs)
                
                output_dict = dict(
                    predicted_forcings=('pred_', predicted_forcings,'_mean'),
                    predicted_std = ('pred_',predicted_std,'_std'),
                    true_forcings = ('true_',true_forcings,'')
                )
                data_vars = {}
                for key,val in output_dict.items():
                    pref,vals,suff = val
                    names = list(vals.data_vars.keys())
                    rename_dict = {nm : pref +nm + suff  for nm in names}
                    vals = vals.rename(rename_dict).isel(lat = 0,lon = 0).drop('lat lon'.split())
                    for name in rename_dict.values():
                        data_vars[name] = vals[name]
                data_vars['lat'] = xr.DataArray(data = [coord[0]],dims = ['coord_id'],coords = dict(
                    coord_id = (['coord_id'],[cid])
                ) )
                data_vars['lon'] = xr.DataArray(data = [coord[1]],dims = ['coord_id'],coords = dict(
                    coord_id = (['coord_id'],[cid])
                ) )
                ds = concat(**data_vars)
                if stats is None:
                    stats = ds
                else:
                    stats = xr.merge([stats,ds])
            nt += 1
            if nt == 400:
                break
            flushed_print('\t\t',nt)
    if not os.path.exists(TIME_LAPSE):
        os.makedirs(TIME_LAPSE)
    filename = os.path.join(TIME_LAPSE,modelid+'.nc')
    print(filename)
    stats.to_netcdf(filename,mode = 'w')


            

            






if __name__=='__main__':
    main()
