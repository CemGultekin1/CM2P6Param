from transforms.coarse_graining import base_transform, gcm_filtering,greedy_coarse_grain, greedy_scipy_filtering, plain_coarse_grain, scipy_filtering
from transforms.coarse_graining_inverse import inverse_filtering, inverse_gcm_filtering, inverse_greedy_scipy_filtering, leaky_inverse_filtering
from transforms.grids import forward_difference
from transforms.krylov import  two_parts_krylov_inversion
from utils.xarray import concat, plot_ds, unbind
import numpy as np
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
        self.inv_filtering : inverse_filtering = self.inv_filtering_class(*args,**kwargs)
    def __call__(self, hres, keys,rename,lres = {},clres = {},\
                             hres0= {},lres0 = {},clres0 = {}):
        forcings,(clres, lres) = super(base_lsrp_subgrid_forcing,self).__call__(hres,keys,rename,lres =  lres,clres = clres)
        # coarse_vars = {key:clres[key] for key in vars.keys()}

        # lres = {x:self.filtering(y,) if x not in lres else lres[x] for x,y in vars.items()}
        # coarse_vars = {key:self.coarse_grain(val)for key,val in lres.items()}
        hres0 = {key:self.inv_filtering(val) if key not in hres0 else hres0[key] for key,val in clres.items() if key in hres}

        forcings_lsrp,(clres0,lres0)= super(base_lsrp_subgrid_forcing,self).__call__(hres0,keys,rename,lres = lres0,clres = clres0)
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
    def decorate(self, call:callable,intype:str = 'wet',outtype = 'wet'):#,terminate_flag :bool = False):
        inmask = self.get_mask(intype)
        outmask = self.get_mask(outtype)
        ds = self.ds.copy()#.fillna(0)
        def __call(x:np.ndarray):
            xx = ds.data
            shp = xx.shape
            xx = xx.reshape([-1,])
            xx[inmask == 1] = x
            xx[inmask == 0] = 0
            xx = xx.reshape(*shp)
            ds.data = xx
            ds1 = call(ds.copy())
            
            # bf = dict(
            #     before = ds,
            #     after = ds1,
            #     diff = ds1 - ds
            # )
            # imname = f'{intype}_{outtype}.png'
            # print(imname)
            # plot_ds(bf,imname,ncols = 3)
            # if terminate_flag:
            #     raise Exception
            xx = ds1.data.reshape([-1])
            return xx[outmask]
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

class krylov_lsrp_subgrid_forcing(base_lsrp_subgrid_forcing):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def __call__(self, hres:dict, keys,rename,lres = {},clres = {},\
                             hres0= {},lres0 = {},clres0 = {}):
        forcings,(clres,lres) = super(base_lsrp_subgrid_forcing,self).__call__(hres,keys,rename,lres = lres,clres = clres)
        def forward(lres):
            y = self.inv_filtering(lres,inverse = True)
            lres1 = self.filtering(y)
            clres = self.coarse_grain(lres1)
            return clres
        def wet2dry(lres):
            return self.inv_filtering(self.inv_filtering(lres,inverse = True),inverse = False)

        dwxr = xr2np_utility(list(clres.values())[0])
        ww =    dwxr.decorate(forward,      intype = 'wet',  outtype = 'wet')
        w2l =   dwxr.decorate(wet2dry,      intype='wet',    outtype = 'dry')
        wl =    dwxr.decorate(forward,      intype = 'dry',  outtype = 'wet')
        
        def run_gmres(u:xr.DataArray):
            solver = two_parts_krylov_inversion(8,1e-2,ww,wl,w2l)
            drywet_separate_solution = solver.solve(dwxr.get_wet_part(u))
            solution = dwxr.merge(*drywet_separate_solution)
            return self.inv_filtering(solution,inverse = True)
        hres0 = {key: run_gmres(val) if key not in hres0 else hres0[key] for key,val in clres.items()}

        
        if 'temp' in keys:
            plot_ds(hres0,'hres0.png')

        forcings_lsrp,(clres0,lres0) = super(base_lsrp_subgrid_forcing,self).__call__(hres0,keys,rename,clres = clres0,lres = lres0)
        forcings_lsrp = {f"{key}_res":  forcings[key] - forcings_lsrp[key] for key in rename}

        forcings = dict(forcings,**forcings_lsrp)

        return forcings,(clres,lres),(clres0,lres0,hres0)
    



class gcm_subgrid_forcing(base_subgrid_forcing):
    filtering_class = gcm_filtering
    coarse_grain_class = greedy_coarse_grain

class scipy_subgrid_forcing(base_subgrid_forcing):
    filtering_class = scipy_filtering
    coarse_grain_class =  plain_coarse_grain

class greedy_scipy_subgrid_forcing(scipy_subgrid_forcing):
    filtering_class = greedy_scipy_filtering
    coarse_grain_class =  greedy_coarse_grain

class greedy_scipy_lsrp_subgrid_forcing(krylov_lsrp_subgrid_forcing):#base_lsrp_subgrid_forcing):#
    filtering_class = greedy_scipy_filtering
    coarse_grain_class =  greedy_coarse_grain
    inv_filtering_class = leaky_inverse_filtering

class gcm_lsrp_subgrid_forcing(base_lsrp_subgrid_forcing):
    filtering_class = gcm_filtering
    coarse_grain_class = greedy_coarse_grain
    inv_filtering_class = inverse_gcm_filtering
        
