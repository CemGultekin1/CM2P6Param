import itertools
from transforms.coarse_graining import base_transform, gcm_filtering,greedy_coarse_grain, plain_coarse_grain, scipy_filtering
from transforms.coarse_graining_inverse import inverse_gcm_filtering
from transforms.grids import forward_difference
from utils.xarray import concat, unbind

import xarray as xr
import numpy as np

class base_subgrid_forcing(base_transform):
    def __init__(self,*args,\
        grid_separation = 'dy dx'.split(),\
        momentum = 'u v'.split(),**kwargs):
        super().__init__(*args,**kwargs)
        self.filtering = None
        self.coarse_grain = None
        self.grid_separation = grid_separation
        self.momentum = momentum
    def compute_flux(self,hresdict:dict,):
        '''
        Takes high resolution U-grid variables in dictionary uvars and T-grid variables in dictionary tvars
        Takes their fine-grid derivatives across latitude and longitude
        Returns the fine-grid objects and their coarse-grid counterparts and their coarse-grid derivatives across latitude and longitude 
        '''
        dlat = {f"dlat_{x}":forward_difference(y,self.grid[self.grid_separation[0]],self.dims[0]) for x,y in hresdict.items()}
        dlon = {f"dlon_{x}":forward_difference(y,self.grid[self.grid_separation[1]],self.dims[1]) for x,y in hresdict.items()}
        hres_flux = dict(**dlat,**dlon)
        return hres_flux
    def __call__(self,hres,keys,rename):
        lres = {x:self.filtering(y) for x,y in hres.items()}
        hres_flux = self.compute_flux({key:hres[key] for key in keys})
        lres_flux = self.compute_flux({key:lres[key] for key in keys})

        forcings =  dict(lres,**{
            rn : self._subgrid_forcing_formula(hres,lres,hres_flux,lres_flux,key) for key,rn in zip(keys,rename)
        })
        return concat(**{
            key:self.coarse_grain(x) for key,x in forcings.items()
        })


    def _subgrid_forcing_formula(self,hresvars,lresvars,hres_flux,lres_flux,key):
        u :str= self.momentum[0]
        v :str= self.momentum[1]
        adv1 = hresvars[u]*hres_flux[f"dlon_{key}"] + hresvars[v]*hres_flux[f"dlat_{key}"]
        adv1 = self.filtering(adv1)
        adv2 = lresvars[u]*lres_flux[f"dlon_{key}"] + lresvars[v]*lres_flux[f"dlat_{key}"]
        return  adv2 - adv1


class base_lsrp_subgrid_forcing(base_subgrid_forcing):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.inv_filtering = None
        self._wet_density =  None
    def __call__(self, vars, keys,rename):
        forcings = unbind(super(base_lsrp_subgrid_forcing,self).__call__(vars,keys,rename))
        coarse_vars = {key:forcings[key] for key in vars.keys()}
        lres = {x:self.filtering(y,) for x,y in vars.items()}
        coarse_vars = {key:self.coarse_grain(val)for key,val in lres.items()}
        vars0 = {key:self.inv_filtering(val) for key,val in coarse_vars.items()}
        forcings_lsrp = unbind(super(base_lsrp_subgrid_forcing,self).__call__(vars0,keys,rename))
        forcings_lsrp = {f"{key}_res":  forcings[key] - forcings_lsrp[key] for key in rename}
        # forcings_lsrp["wet_density_res"] = self._wet_density.copy()
        forcings = dict(forcings,**forcings_lsrp)
        return concat(**forcings)

    



class gcm_subgrid_forcing(base_subgrid_forcing,):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.filtering = gcm_filtering(*args,**kwargs)
        self.coarse_grain = greedy_coarse_grain(*args,**kwargs)

class scipy_subgrid_forcing(base_subgrid_forcing,):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.filtering = scipy_filtering(*args,**kwargs)
        self.coarse_grain = plain_coarse_grain(*args,**kwargs)


class gcm_lsrp_subgrid_forcing(base_lsrp_subgrid_forcing,gcm_subgrid_forcing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inv_filtering = inverse_gcm_filtering(*args,**kwargs)
        # self._wet_density = prepare_wet_density(self.inv_filtering._full_wet_density,self.dims,sp = 2)
        


# def prepare_wet_density(wet_density,dims,sp = 1,thresh = 0.99):
#     return xr.where(wet_density > thresh, wet_density,0)
#     land_density = 1 - wet_density
#     cland = land_density*0
#     t = 0
#     for i in itertools.product(range(-sp,sp+1),range(-sp,sp+1)):
#         cland += land_density.roll({d:i_ for d,i_ in zip(dims,i)})
#         t += 1
#     cland = cland/t
#     return xr.where(cland < 1 - thresh, 1- cland, 0)