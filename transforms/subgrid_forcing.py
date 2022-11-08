from typing import Callable
# from transforms.coarse_grain_inversion import coarse_grain_projection
# from transforms.grids import forward_difference#, get_grid_vars, ugrid2tgrid
# from utils.slurm import flushed_print
import xarray as xr
from utils.xarray import concat


def _subgrid_forcing(hres_left,lres_left,hres_right,lres_right,key,coarse_grain,):
    adv1 = hres_left['u']*hres_right[f"dlon_{key}"] + hres_left['v']*hres_right[f"dlat_{key}"]
    adv1 = coarse_grain(adv1)
    
    adv2 = lres_left['u']*lres_right[f"dlon_{key}"] + lres_left['v']*lres_right[f"dlat_{key}"]
   
    return  adv1 - adv2

# def _possibly_deep_subgrid_forcing(hres_left,lres_left,hres_right,lres_right,key,coarse_grain):
#     left_deep  = 'depth' in hres_left.coords
#     right_deep = 'depth' in hres_right.coords
#     matching_depths = left_deep and right_deep
#     def sel_depth(i,*xs):
#         return [{key: x[key].isel(depth = i) for key in x} for x in xs]

#     if left_deep:
#         ndepth = len(hres_left.depth)
#     elif right_deep:
#         ndepth = len(hres_right.depth)
#     if left_deep or right_deep:
#         sfs = []
#         for i in range(ndepth):
#             if left_deep:
#                 _hres_left,_lres_left = sel_depth(i,hres_left,lres_left)
#             else:
#                 _hres_left,_lres_left = hres_left,lres_left
#             if right_deep:
#                 _hres_right,_lres_right = sel_depth(i,hres_right,lres_right)
#             else:
#                 _hres_right,_lres_right = hres_right,lres_right
#             sbg = _subgrid_forcing(_hres_left,_lres_left,_hres_right,_lres_right,key,coarse_grain,)
#             sbg.
        
            
    
def subgrid_forcing(hres_left,lres_left,hres_right,lres_right,root_names,target_names,coarse_grain:Callable):
    '''
    :u,v,T: high resolution variables U-grid and T-grid
    :coarse_grain_u,coarse_grain_t: coarse graining methods for U-grid and T-grid separately
    '''
    ugridforcings = {tgt: _subgrid_forcing(hres_left,lres_left,hres_right,lres_right,key,coarse_grain,) for tgt,key in zip(target_names,root_names)}#"u v".split()}

    return concat(**ugridforcings)

