from typing import Callable
import xarray as xr
from utils.xarray import concat


def _subgrid_forcing_formula(hres_left,lres_left,hres_right,lres_right,key,coarse_grain,):
    adv1 = hres_left['u']*hres_right[f"dlon_{key}"] + hres_left['v']*hres_right[f"dlat_{key}"]
    adv1 = coarse_grain(adv1)
    adv2 = lres_left['u']*lres_right[f"dlon_{key}"] + lres_left['v']*lres_right[f"dlat_{key}"]
    return  adv2 - adv1

            
    
def subgrid_forcing(hres_left,lres_left,hres_right,lres_right,root_names,target_names,coarse_grain:Callable):
    '''
    Takes fine-grid (hres) and coarse-grid (lres) variables to be placed into right and left positions of subgrid forcing
    The quantities should all be on the same grid as :coarse_grain: function is defined on. 
    return xr.Dataset with subgrid forcings computed for variables names :root_names:
    the names are changed to :target_names:
    the returned dataset has 'lat' and 'lon' as coordinate names, removing the distinction between U and T-grid
    '''
    ugridforcings = {tgt: _subgrid_forcing_formula(hres_left,lres_left,hres_right,lres_right,key,coarse_grain,) for tgt,key in zip(target_names,root_names)}

    return concat(**ugridforcings)

