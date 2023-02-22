from transforms.coarse_graining import base_transform, gcm_filtering,greedy_coarse_grain, greedy_scipy_filtering, plain_coarse_grain, scipy_filtering
from transforms.coarse_graining_inverse import  matmult_masked_filtering,matmult_filtering
from transforms.grids import forward_difference
from transforms.krylov import  krylov_inversion
import numpy as np
from utils.xarray import plot_ds
import xarray as xr


class base_subgrid_forcing(base_transform):
    filtering_class = None
    coarse_grain_class = None
    def __init__(self,*args,\
        grid_separation = 'dy dx'.split(),\
        momentum = 'u v'.split(),**kwargs):
        super().__init__(*args,**kwargs)
        self.filtering = self.filtering_class(*args,**kwargs)
        self.coarse_grain = self.coarse_grain_class(*args,**kwargs)
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
    def __call__(self,hres,keys,rename,lres = {},clres = {}):
        lres = {x:self.filtering(y) if x not in lres else lres[x] for x,y in hres.items()}
        hres_flux = self.compute_flux({key:hres[key] for key in keys})
        lres_flux = self.compute_flux({key:lres[key] for key in keys})

        forcings =  { rn : self._subgrid_forcing_formula(hres,lres,hres_flux,lres_flux,key) for key,rn in zip(keys,rename) }
        clres = {key:self.coarse_grain(x) if key not in clres else clres[key] for key,x in lres.items()}
        forcings = {key:self.coarse_grain(x) for key,x in forcings.items()}
        return forcings,(clres,lres)


    def _subgrid_forcing_formula(self,hresvars,lresvars,hres_flux,lres_flux,key):
        u :str= self.momentum[0]
        v :str= self.momentum[1]
        adv1 = hresvars[u]*hres_flux[f"dlon_{key}"] + hresvars[v]*hres_flux[f"dlat_{key}"]
        adv1 = self.filtering(adv1)
        adv2 = lresvars[u]*lres_flux[f"dlon_{key}"] + lresvars[v]*lres_flux[f"dlat_{key}"]
        return  adv2 - adv1


class base_lsrp_subgrid_forcing(base_subgrid_forcing):
    inv_filtering_class = None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.inv_filtering : matmult_filtering = self.inv_filtering_class(*args,**kwargs)
    def __call__(self, hres, keys,rename,lres = {},clres = {},\
                             hres0= {},lres0 = {},clres0 = {}):
        forcings,(clres, lres) = super().__call__(hres,keys,rename,lres =  lres,clres = clres)
        hres0 = {key:self.inv_filtering(val) if key not in hres0 else hres0[key] for key,val in clres.items() if key in hres}
        
        forcings_lsrp,(clres0,lres0)= super().__call__(hres0,keys,rename,lres = lres0,clres = clres0)
        forcings_lsrp = {f"{key}_res":  forcings[key] - forcings_lsrp[key] for key in rename}
        forcings = dict(forcings,**forcings_lsrp)
        return forcings,(clres,lres),(clres0,lres0,hres0)




class xr2np_utility:
    def __init__(self,ds,) -> None:
        self.ds = ds.copy()
        self.flat_wetpoints = (1 - np.isnan(self.ds.data.reshape([-1]))).astype(bool)
    def get_mask(self, parts:str):
        if 'wet' == parts:
            mask = self.flat_wetpoints
        else:
            assert 'dry' == parts
            mask = 1 - self.flat_wetpoints
        return mask.astype(bool)
    def get_wet_part(self,ds:xr.DataArray):
        data = ds.data.reshape([-1])
        return data[self.flat_wetpoints]
    def zero_wet_part(self,ds:xr.DataArray):
        shp = ds.data.shape
        data = ds.data.reshape([-1])
        data[self.flat_wetpoints] = 0
        ds.data = data.reshape(shp)
        return ds
    def np2xr(self,x:np.ndarray):
        ds = self.ds.copy()
        shp = ds.data.shape
        ds.data = x.reshape(shp)
        return ds
    def decorate(self, call:callable,):#,terminate_flag :bool = False):
        ds = self.ds.copy()#.fillna(0)
        shp = ds.data.shape
        def __call(x:np.ndarray):
            ds.data = x.reshape(shp)
            ds1 = call(ds.fillna(0))
            return ds1.data.reshape([-1])
        return __call
    def merge(self,wetx:np.ndarray,dryx:np.ndarray):
        ds = self.ds.copy()
        xx = ds.data
        shp = xx.shape
        xx = xx.reshape([-1,])
        xx[self.flat_wetpoints == 1] = wetx
        xx[self.flat_wetpoints == 0] = dryx
        xx = xx.reshape(*shp)
        ds.data = xx
        return ds

class landfilling_krylov_lsrp_subgrid_forcing(base_lsrp_subgrid_forcing):
    def __call__(self, hres:dict, keys,rename,lres = {},clres = {},\
                             hres0= {},lres0 = {},clres0 = {}):
        forcings,(clres,lres) = super(base_lsrp_subgrid_forcing,self).__call__(hres,keys,rename,lres = lres,clres = clres)

        dwxr = xr2np_utility(list(clres.values())[0])
        def orthproj(lres):
            return lres - self.inv_filtering(self.inv_filtering(lres,inverse = True),inverse = False)
        def matmultip(lres):
            zwet = dwxr.zero_wet_part(lres).fillna(0)
            return orthproj(zwet).fillna(0)
        decorated_matmultip = dwxr.decorate(matmultip)
        def run_gmres(u:xr.DataArray):
            u = u.fillna(0)
            
            orthu = orthproj(u)

            # plot_ds({'orthu':orthu},'orthu.png',ncols = 1)
            # raise Exception
            solver = krylov_inversion(8,1e-2,decorated_matmultip)
            landfill_np = solver.solve( - orthu.fillna(0).values.reshape([-1]))
            landfill_u = dwxr.zero_wet_part(dwxr.np2xr(landfill_np))
            filled_u =  u + landfill_u
            return self.inv_filtering(filled_u,inverse = True)
        hres0 = {key: run_gmres(val) if key not in hres0 else hres0[key] for key,val in clres.items()}
        # landfills = {key + '_landfill':run_gmres(val) for key,val in clres.items()}

        # plot_ds(dict(clres,**landfills),'landfills.png')
        # raise Exception
        
        # if 'temp' in keys:
        #     plot_ds(hres0,'hres0.png')

        forcings_lsrp,(clres0,lres0) = super(base_lsrp_subgrid_forcing,self).__call__(hres0,keys,rename,clres = clres0,lres = lres0)
        forcings_lsrp = {f"{key}_res":  forcings[key] - forcings_lsrp[key] for key in rename}

        forcings = dict(forcings,**forcings_lsrp)

        return forcings,(clres,lres),(clres0,lres0,hres0)
    



# class gcm_subgrid_forcing(base_subgrid_forcing):
#     filtering_class = gcm_filtering
#     coarse_grain_class = greedy_coarse_grain

class scipy_subgrid_forcing(base_subgrid_forcing):
    filtering_class = scipy_filtering
    coarse_grain_class =  plain_coarse_grain

# class greedy_scipy_subgrid_forcing(scipy_subgrid_forcing):
#     filtering_class = greedy_scipy_filtering
#     coarse_grain_class =  greedy_coarse_grain





class scipy_lsrp_subgrid_forcing(landfilling_krylov_lsrp_subgrid_forcing):
    filtering_class = scipy_filtering
    coarse_grain_class =  plain_coarse_grain
    inv_filtering_class = matmult_filtering


class greedy_scipy_lsrp_subgrid_forcing(landfilling_krylov_lsrp_subgrid_forcing):
    filtering_class = greedy_scipy_filtering
    coarse_grain_class =  greedy_coarse_grain
    inv_filtering_class = matmult_masked_filtering

class gcm_lsrp_subgrid_forcing(landfilling_krylov_lsrp_subgrid_forcing):
    filtering_class = gcm_filtering
    coarse_grain_class = greedy_coarse_grain
    inv_filtering_class = matmult_masked_filtering



        
filtering_classes = {
    "gcm":gcm_lsrp_subgrid_forcing,\
    "gaussian":scipy_lsrp_subgrid_forcing,\
    "greedy_gaussian":greedy_scipy_lsrp_subgrid_forcing
}