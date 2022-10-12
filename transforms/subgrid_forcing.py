from typing import Callable
import xarray as xr
import numpy as np
from utils.xarray import concat

def forward_difference(x:xr.DataArray,field):
    dx = x.diff(field)/x[field].diff(field)
    f0 = x[field][0]
    dx = dx.pad({field : (1,0)},constant_values = np.nan)
    dxf = dx[field].values
    dxf[0] = f0
    dx[field] = dxf
    return dx

def hreslres(u,v,T,coarse_grain_u,coarse_grain_t):
    '''
    converts hres :u,v,t: into lres versions by coarse-graining
    including their derivatives across latitude and longitude
    '''
    coarse_grain = lambda x, tgrid: coarse_grain_u(x) if not tgrid else coarse_grain_t(x)

    hres = dict(u=u,v=v,T=T)
    lres = {x:coarse_grain(y, 'T' in x) for x,y in hres.items()}

    dlat = {f"d{x}dlat":forward_difference(y,"lat") for x,y in hres.items()}
    dlon = {f"d{x}dlon":forward_difference(y,"lon") for x,y in hres.items()}
    hres = dict(hres,**dlat,**dlon)

    dlat = {f"d{x}dlat":forward_difference(y,"lat") for x,y in lres.items()}
    dlon = {f"d{x}dlon":forward_difference(y,"lon") for x,y in lres.items()}
    lres = dict(lres,**dlat,**dlon)
    
    for res in (hres,lres):
        for key in res.keys():
            if "T" in key:
                res[key] = res[key].interp(lon = res["u"].lon,lat = res["u"].lat)
    return hres,lres

def _subgrid_forcing(hres,lres,key,coarse_grain,):
    adv1 = hres["u"]*hres[f"d{key}dlon"] + hres["v"]*hres[f"d{key}dlat"]
    adv1 = coarse_grain(adv1)
    adv2 = lres["u"]*lres[f"d{key}dlon"] + lres["v"]*lres[f"d{key}dlat"]
    return  adv2 - adv1
def subgrid_forcing(u:xr.DataArray,v:xr.DataArray,T:xr.DataArray,coarse_grain_u:Callable,coarse_grain_t:Callable):
    '''
    :u,v,T: high resolution variables U-grid and T-grid
    :coarse_grain_u,coarse_grain_t: coarse graining methods for U-grid and T-grid separately
    '''
    hres, lres = hreslres(u,v,T,coarse_grain_u,coarse_grain_t)
    S = concat(**{key: _subgrid_forcing(hres,lres,key,coarse_grain_u,) for key in "u v T".split()})
    U = concat(**{key:val for key,val in lres.items() if "dlon" not in key and "dlat" not in key})
    return S,U