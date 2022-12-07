import itertools
import os
import sys
from data.exceptions import RequestDoesntExist
from plots.metrics import metrics_dataset, moments_dataset
from run.train import Timer
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
from utils.xarray import fromtensor, fromtorchdict, fromtorchdict2tensor, plot_ds
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
def lsrp_pred(respred,tr):
    keys= list(respred.data_vars.keys())
    data_vars = {}
    coords = {key:val for key,val in tr.coords.items()}
    for key in  keys:
        trkey = key.replace('0_res','').replace('1_res','')
        err = tr[trkey] - tr[key]
        data_vars[trkey] = (err.dims,err.values)
        respred[key] = err + respred[key]
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


def main():
    args = sys.argv[1:]
    runargs,_ = options(args,key = "run")

    modelid,_,net,_,_,_,_,runargs=load_model(args)
    device = get_device()
    lsrp_flag, lsrpid = get_lsrp_modelid(args)
    
    kwargs = dict(contained = '' if not lsrp_flag else 'res')
    assert runargs.mode == "eval"
    # multidatargs = populate_data_options(args,non_static_params=['depth','co2'],domain = 'global')
    multidatargs = [args]
    allstats = {}
    for datargs in multidatargs:
        try:
            test_generator, = get_data(datargs,half_spread = net.spread, torch_flag = False, data_loaders = True,groups = ('test',))
        except RequestDoesntExist:
            print('data not found!')
            test_generator = None
        if test_generator is None:
            continue
        stats = {}
        nt = 0
        # timer = Timer()
        for fields,forcings,forcing_mask,_,forcing_coords in test_generator:
            fields_tensor = fromtorchdict2tensor(fields).type(torch.float32)
            depth = forcing_coords['depth'].item()
            co2 = forcing_coords['co2'].item()
            kwargs = dict(contained = '' if not lsrp_flag else 'res', \
                expand_dims = {'co2':[co2],'depth':[depth]})
            if nt ==  0:
                flushed_print(depth,co2)

            with torch.set_grad_enabled(False):
                mean,_ =  net.forward(fields_tensor.to(device))
                mean = mean.to("cpu")

            
            # outfields = fromtorchdict2tensor(forcings).type(torch.float32)
            # mask = fromtorchdict2tensor(forcing_mask).type(torch.float32)
            # yhat = mean.numpy()[0]
            # y = outfields.numpy()[0]
            # m = mask.numpy()[0] < 0.5
            # yhat[m] = np.nan
            # y[m] = np.nan
            # prst = lambda y: print(np.mean(y[y==y]),np.std(y[y==y]))
            # prst(y),prst(yhat),prst(fields_tensor.numpy())
            # nchan = yhat.shape[0]
            # import matplotlib.pyplot as plt
            # fig,axs = plt.subplots(nchan,2,figsize = (2*5,nchan*6))
            # for chani in range(nchan):
            #     ax = axs[chani,0]
            #     ax.imshow(y[chani,::-1])
            #     ax = axs[chani,1]
            #     ax.imshow(yhat[chani,::-1])
            # fig.savefig('eval_intervention.png')
            # return


            predicted_forcings = fromtensor(mean,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
            true_forcings = fromtorchdict(forcings,forcing_coords,forcing_mask,denormalize = True,**kwargs)
            if lsrp_flag:
                predicted_forcings,true_forcings = lsrp_pred(predicted_forcings,true_forcings)
                predicted_forcings,lsrp_forcings = predicted_forcings
                stats = update_stats(stats,lsrp_forcings,true_forcings,lsrpid)
            
            # err = np.log10(np.abs(true_forcings - predicted_forcings))
            # plot_ds(predicted_forcings,'predicted_forcings',ncols = 1)
            # plot_ds(true_forcings,'true_forcings',ncols = 1)           
            # plot_ds(err,'err',ncols = 1)
            

            # return
            stats = update_stats(stats,predicted_forcings,true_forcings,modelid)
            nt += 1

            # break
            # if nt == 16:
            # for key,stats_ in stats.items():
            #     stats__ = metrics_dataset(stats_/nt,reduckwargs = {})
            #     names = list(stats__.data_vars.keys())
            #     ncols = 2
            #     nrows  = int(np.ceil(len(names)/ncols))
                
            #     fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*5,nrows*5))
            #     for z,(i,j)  in enumerate(itertools.product(range(nrows),range(ncols))):
            #         ax = axs[i,j]
            #         kwargs = dict(vmin =0.5,vmax = 1) if 'r2' in names[z] else dict()
            #         kwargs = dict(vmin =-1,vmax = 1) if 'corr' in names[z] else kwargs
            #         stats__[names[z]].isel(co2 = 0,depth = 0).plot(ax = ax,**kwargs)
            #         ax.set_title(names[z])
                    
            #     fig.savefig(f'_{nt}_{key}.png')
            #     plt.close()
            # if nt == 16:
            #     return
            #     return
            # break

        for key in stats:
            stats[key] = stats[key]/nt
            if key not in allstats:
                allstats[key] = []
            allstats[key].append(stats[key].copy())

    for key in allstats:
        filename = os.path.join(EVALS,key+'.nc')
        print(filename)
        xr.merge(allstats[key]).to_netcdf(filename,mode = 'w')


            

            






if __name__=='__main__':
    main()
