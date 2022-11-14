import numpy as np
import xarray as xr

def metrics_dataset(sn,reduckwargs = dict(dim = ['lat','lon'])):
    dvn = list(sn.data_vars)
    dvn = np.unique([dn.split('_')[0] for dn in dvn])
    xarrs = []
    for key in dvn:
        tmom1 = sn[f"{key}_true_mom1"]
        pmom1 = sn[f"{key}_pred_mom1"]
        tmom2 = sn[f"{key}_true_mom2"]
        pmom2 = sn[f"{key}_pred_mom2"]
        cmom2 = sn[f"{key}_cross"]

        if len(reduckwargs) > 0:
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
        r2.name = f"{key}_r2"
        mse.name = f"{key}_mse"
        sc2.name = f"{key}_sc2"
        correlation.name = f"{key}_corr"
        xarrs.extend([r2,correlation,mse,sc2])
    return xr.merge(xarrs)
