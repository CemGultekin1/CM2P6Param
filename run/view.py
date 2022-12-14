import os
import random
import sys
from data.exceptions import RequestDoesntExist
from run.eval import lsrp_pred
import torch
from data.load import get_data
from data.vars import get_var_mask_name
from models.load import load_model
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
import numpy as np
from utils.paths import VIEWS
from utils.xarray import fromtensor, fromtorchdict, fromtorchdict2tensor
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

def torch_stack(cdict):
    torchtensor = torch.stack([cdict[key]['val'] for key in cdict],dim= 1)
    return torchtensor

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
def to_xarray(torchdict,depth,time,itime):
    data_vars = {
        key : (["itime","depth","lat","lon"] ,np.stack([torchdict[key]['val'].numpy()],axis= 0 )) for key in torchdict
    }
    data_vars['time'] = (["itime","depth"],time.reshape(1,1) )
    for key in torchdict:
        lat = torchdict[key]["lat"][0,:].numpy()
        lon = torchdict[key]["lon"][0,:].numpy()
        break
    coords = dict(lat = (["lat"],lat),lon = (["lon"],lon),\
        depth = (["depth"],depth.reshape([-1])), itime = (["itime"],itime.reshape([-1])))
    return xr.Dataset(data_vars = data_vars,coords = coords)

def err_scale_dataset(mean,truef):
    err = mean
    sc2 = truef

    names = list(err.data_vars)
    for name in names:
        err = err.rename({name : name})
        if name != 'time':
            sc2 = sc2.rename({name : name+'_true'})
    return xr.merge([err,sc2])

def ifnotnone_merge(evs,part_evs):
    if evs is None:
        evs = part_evs
    else:
        evs = xr.merge([evs,part_evs])
    return evs

def main():
    # def set_seed():
    #     seed = 0
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     np.random.seed(seed)
    #     random.seed(seed)
    args = sys.argv[1:]
    
    modelid,_,net,_,_,_,_,runargs=load_model(args)
    device = get_device()
    net.to(device)
    
    runargs,_ = options(args,key = "run")
    lsrp_flag = runargs.lsrp
    lsrpid = f'lsrp_{lsrp_flag}'
    assert runargs.mode == "view"
    
    multidatargs = populate_data_options(args,non_static_params=[])#"depth"])
    allstats = []
    for datargs in multidatargs:
        try:
            test_generator, = get_data(datargs,half_spread = net.spread, torch_flag = False, data_loaders = True,groups = ('train',))
        except RequestDoesntExist:
            print('data not found!')
            test_generator = None
        if test_generator is None:
            continue
        nt = 0
        nt_limit = 5
        
        for fields,forcings,field_mask,forcing_mask,field_coords,forcing_coords in test_generator:
            time,depth,co2 = field_coords['time'].item(),field_coords['depth'].item(),field_coords['co2'].item()
            print(time,depth,co2)
            kwargs = dict(contained = '' if not lsrp_flag else 'res', \
                expand_dims = {'co2':[co2],'time':[time],'depth':[depth]})
            fields_tensor = fromtorchdict2tensor(fields).type(torch.float32)
            # mean = fromtorchdict2tensor(forcings,**kwargs).type(torch.float32)
            
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
            # fig.savefig('view_intervention.png')
            # return



            predicted_forcings = fromtensor(mean,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
            true_forcings = fromtorchdict(forcings,forcing_coords,forcing_mask,denormalize = True,**kwargs)
            true_fields = fromtorchdict(fields,field_coords,field_mask,denormalize = True,**kwargs)

            if lsrp_flag:
                (predicted_forcings,lsrp_forcings),true_forcings = lsrp_pred(predicted_forcings,true_forcings)
            def rename_dict(suffix):
                renames = {}
                for key in true_forcings.data_vars.keys():
                    renames[key] = f'{key}_{suffix}'
                return renames

            err_forcings = true_forcings - predicted_forcings
            err_forcings = err_forcings.rename(rename_dict('err'))
            true_forcings = true_forcings.rename(rename_dict('true'))
            predictions_ = xr.merge([predicted_forcings,err_forcings,true_forcings,true_fields])
            if lsrp_flag:
                err_forcings = true_forcings - lsrp_forcings
                err_forcings = err_forcings.rename(rename_dict('err'))
                lsrp_predictions_ = xr.merge([lsrp_forcings,err_forcings,true_forcings,true_fields])
                allstats.append({lsrpid:lsrp_predictions_})
            allstats.append({modelid:predictions_})

            nt+=1
            if nt == nt_limit:
                break
        
    evs = {modelid:xr.merge([alst[modelid] for alst in allstats])}
    if lsrp_flag:
        evs[lsrpid] = xr.merge([alst[lsrpid] for alst in allstats])
    for key in evs:
        fileame = os.path.join(VIEWS,key+'.nc')
        evs[key].sel(lon = slice(-180,180),).to_netcdf(fileame,mode = 'w')

            

            






if __name__=='__main__':
    main()
