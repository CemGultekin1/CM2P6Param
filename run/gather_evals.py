import os
from utils.paths import SLURM, EVALS, all_eval_path
import xarray as xr
from utils.arguments import options
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
def getr2(sn,key):
    tmom1 = sn[f"{key}_true_mom1"]
    pmom1 = sn[f"{key}_pred_mom1"]
    tmom2 = sn[f"{key}_true_mom2"]
    pmom2 = sn[f"{key}_pred_mom2"]
    cmom2 = sn[f"{key}_cross"]
    reduckwargs = dict(dim = ['lat','lon'])
    nonan = xr.where(np.isnan(tmom1),0,1)
    def skipna_average(st):
        return xr.where(np.isnan(st),0,st).sum(**reduckwargs)/nonan.sum(**reduckwargs)
    tmom1 = skipna_average(tmom1)
    pmom1 = skipna_average(pmom1)
    tmom2 = skipna_average(tmom2)
    pmom2 = skipna_average(pmom2)
    cmom2 = skipna_average(cmom2)


    mse = tmom2 + pmom2 - 2*cmom2
    sc2 = tmom2
    
    tvar = tmom2 - np.square(tmom1)
    pvar = pmom2 - np.square(pmom1)

    r2 = 1 - mse/sc2
    correlation = (cmom2 - pmom1*tmom1)/np.sqrt(tvar*pvar)
    return dict(r2 = r2,correlation = correlation)
    

def append_statistics(stats,sn:xr.Dataset,coordvals):
    # reduckwargs = dict(dim = ["lat","lon"],skipna = True)
    model = {}
    for key in 'Su Sv ST'.split():
        if f"{key}_true_mom1" not in sn.data_vars.keys():
            continue
        for _key,_val in getr2(sn,key).items():
            model[f"{key}_{_key}"] = _val
    for key,val in model.items():
        val.name = key
        model[key] = val
    modelev = xr.merge(list(model.values()))
    for cname,cval in coordvals.items():
        modelev = modelev.expand_dims(dim = {cname:cval},axis= 0)
    if stats is None:
        stats= modelev
    else:
        stats = xr.merge([stats,modelev])
    return stats

def main():
    root = EVALS
    models = os.path.join(SLURM,'evaljob.txt')
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()


    lines = lines + turn_to_lsrp_models(lines)
    coords = ['sigma','temperature','domain','latitude','lsrp','depth','seed','model']
    coordnames = ['sigma','temperature','training_domain','latitude_features','lsrp','training_depth','seed','model']
    stats = None
    for line in lines:
        modelargs,modelid = options(line.split(),key = "model")
        coordvals = {}
        for cn,coord in zip(coordnames,coords):
            val = modelargs.__getattribute__(coord)
            if isinstance(val,bool):
                val = int(val)
            coordvals[cn] = [val]
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            continue
        sn = xr.open_dataset(snfile)
        stats = append_statistics(stats,sn,coordvals)
   
    stats.to_netcdf(all_eval_path(),mode = 'w')
    print(stats)

if __name__=='__main__':
    main()