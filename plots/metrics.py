import numpy as np
import xarray as xr

moment_names = {
    (1,0): 'true_mom1',
    (0,1): 'pred_mom1',
    (2,0): 'true_mom2',
    (0,2): 'pred_mom2',
    (1,1): 'cross'
}

def moments_dataset(tr,prd):
    evals = {}
    evals[(1,0)] = tr.copy()
    evals[(2,0)] = np.square(tr).copy()
    evals[(0,1)] = prd.copy()
    evals[(0,2)] = np.square(prd).copy()
    evals[(1,1)] = (tr*prd).copy()
    evals = {moment_names[key]:val for key,val in evals.items()}
    for ev,val in evals.items():
        evals[ev] = val.rename({key:f'{key}_{ev}' for key in val.data_vars})
    return xr.merge(list(evals.values()))
def metrics_dataset(sn, dim = ['lat','lon']):
    reduckwargs = dict(dim = dim)
    dvn = list(sn.data_vars)
    dvn = np.unique([dn.split('_')[0] for dn in dvn])
    xarrs = []
    for key in dvn:
        moms = {}
        for mtuple,mname in moment_names.items():
            moms[mtuple] = sn[f"{key}_{mname}"].copy()
        if len(reduckwargs) > 0:
            nonan = xr.where(np.isnan(moms[(1,0)]),0,1)
            def skipna_average(st):
                return xr.where(np.isnan(st),0,st).sum(**reduckwargs)/nonan.sum(**reduckwargs)
            moms = {key:skipna_average(val) for key,val in moms.items()}


        mse = moms[(2,0)] + moms[(0,2)] - 2*moms[(1,1)]
        sc2 = moms[(0,2)] 
        
        pvar = moms[(2,0)] - np.square(moms[(1,0)])
        tvar = moms[(0,2)] - np.square(moms[(0,1)])

        r2 = 1 - mse/sc2
        correlation = (moms[(1,1)] - moms[(1,0)]*moms[(0,1)])/np.sqrt(tvar*pvar)
        r2.name = f"{key}_r2"
        mse.name = f"{key}_mse"
        sc2.name = f"{key}_sc2"
        correlation.name = f"{key}_corr"
        xarrs.extend([r2,correlation,mse,sc2])
    return xr.merge(xarrs)
