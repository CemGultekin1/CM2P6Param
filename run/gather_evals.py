import itertools
import os
from plots.metrics import metrics_dataset
from utils.paths import SLURM, EVALS, all_eval_path
from utils.slurm import flushed_print
from utils.xarray import plot_ds, skipna_mean
import xarray as xr
from utils.arguments import args2dict, options
import numpy as np

def get_lsrp_modelid(args):
    runargs,_ = options(args,key = "model")
    lsrp_flag = runargs.lsrp > 0 and runargs.temperature
    if not lsrp_flag:
        return False, None,None
    keys = ['model','sigma']
    vals = [runargs.__getattribute__(key) for key in keys]
    lsrpid = runargs.lsrp - 1
    vals[0] = f'lsrp:{lsrpid}'
    line =' '.join([f'--{k} {v}' for k,v in zip(keys,vals)])
    _,lsrpid = options(line.split(),key = "model")
    return True, lsrpid,line
def turn_to_lsrp_models(lines):
    lsrplines = []
    for i in range(len(lines)):
        line = lines[i]
        lsrp_flag,_,lsrpline = get_lsrp_modelid(line.split())
        if lsrp_flag:
            lsrplines.append(lsrpline)
    lsrplines = np.unique(lsrplines).tolist()
    return lsrplines 

def append_statistics(sn:xr.Dataset,coordvals):
    modelev = metrics_dataset(sn.sel(lat = slice(-85,85)),dim = [])
    # print(modelev)
    # raise Exception
    modelev = skipna_mean(modelev,dim = ['lat','lon'])
    # print(modelev)

    # raise Exception
    # modelev = modelev.fillna(0).mean()
    for c,v in coordvals.items():
        if c not in modelev.coords:
            modelev = modelev.expand_dims(dim = {c:v})
    for key in 'Su Sv Stemp'.split():
        r2key = f"{key}_r2"
        msekey = f"{key}_mse"
        sc2key = f"{key}_sc2"
        if r2key not in modelev.data_vars:
            continue
        modelev[r2key] = 1 - modelev[msekey]/modelev[sc2key]
    # print(modelev.Su_r2.values.item(),1- modelev.Su_mse.values.item()/modelev.Su_sc2.values.item())
    return modelev
def merge_and_save(stats):
    xr.merge(list(stats.values())).to_netcdf(all_eval_path(),mode = 'w')
def main():
    root = EVALS
    models = os.path.join(SLURM,'trainjob.txt')
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()


    lines = lines + turn_to_lsrp_models(lines)
    coords = ['sigma','temperature','domain','latitude','lsrp','depth','seed','model']
    rename = dict(depth = 'training_depth')
    data = {}
    coord = {}
    for i,line in enumerate(lines):
        coordvals,(_,modelid) = args2dict(line.split(),key = 'model',coords = coords)
        for rn,val in rename.items():
            coordvals[val] = coordvals.pop(rn)
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            print(snfile)
            continue
        try:
            sn = xr.open_dataset(snfile)
        except:
            continue
        data[modelid] = append_statistics(sn,coordvals)
        flushed_print(i,snfile.split('/')[-1])
        # if i == 32:
        #     break
    merged_coord = {}
    for ds in data.values():
        for key,val in ds.coords.items():
            if key not in merged_coord:
                merged_coord[key] = []
            merged_coord[key].extend(val.values.tolist())
            merged_coord[key] = np.unique(merged_coord[key]).tolist()
    # print(merged_coord)
    shape = [len(v) for v in merged_coord.values()]
    def empty_arr():
        return np.ones(np.prod(shape))*np.nan
    data_vars = {}
    for modelid,ds in data.items():
        loc_coord = {key:val.values for key,val in ds.coords.items()}
        lkeys = list(loc_coord.keys())
        for valc in itertools.product(*loc_coord.values()):
            # print({k:v for k,v in zip(lkeys,valc)})
            inds = tuple([merged_coord[k].index(v) for k,v in zip(lkeys,valc)])
            alpha = np.ravel_multi_index(inds,shape)
            _ds = ds.sel(**{k:v for k,v in zip(lkeys,valc)}).copy()
            for key in _ds.data_vars.keys():
                if key not in data_vars:
                    data_vars[key] = empty_arr()
                assert np.all(np.isnan(data_vars[key][alpha] ))
                data_vars[key][alpha] = _ds[key].values
                # print(f'data_vars[{key}][{alpha}] = {_ds[key].values}')
    for key,val in data_vars.items():
        data_vars[key] = (list(merged_coord.keys()),val.reshape(shape))
    ds = xr.Dataset(data_vars = data_vars,coords = merged_coord)
    # print(ds.Su_r2.isel(training_depth = 0,model =0,seed = 0,lsrp = 0,latitude = 0,).values.reshape([-1]))
    # return 
   
    ds.to_netcdf(all_eval_path(),mode = 'w')

if __name__=='__main__':
    main()