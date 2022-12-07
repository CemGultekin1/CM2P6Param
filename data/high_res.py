from typing import Callable
from utils.no_torch_xarray import concat, tonumpydict
from utils.xarray import unbind
import xarray as xr
from transforms.grids import get_grid_vars, ugrid2tgrid_interpolation
from transforms.subgrid_forcing import gcm_lsrp_subgrid_forcing
import numpy as np



class HighResCm2p6:
    ds : xr.Dataset
    sigma : int
    half_spread : int
    coarse_grain : Callable
    initiated : bool
    def __init__(self,ds:xr.Dataset,sigma,*args,section = [0,1],**kwargs):
        self.ds = ds.copy()#.isel({f"{prefix}{direction}":slice(1500,2500) for prefix in 'u t'.split() for direction in 'lat lon'.split()})
        self.sigma = sigma
        self.initiated = False
        self.wet_mask = None
        self._ugrid_subgrid_forcing = None
        self._tgrid_subgrid_forcing = None
        self._grid_interpolation = None
        a,b = section
        nt = len(self.ds.time)
        time_secs = np.linspace(0,nt,b+1).astype(int)
        t0 = int(time_secs[a])
        t1 = int(time_secs[a+1])        
        self.ds = self.ds.isel(time = slice(t0,t1))
        self.wet_mask_compute_flag = a == 0

    @property
    def depth(self,):
        return self.ds.depth

    def is_deep(self,):
        return self.depth[0] > 1e-3
    def __len__(self,):
        return len(self.ds.time)*len(self.ds.depth)
    def time_depth_indices(self,i):
        di = i%len(self.ds.depth)
        ti = i//len(self.ds.depth)
        return ti,di

    def get_hres_dataset(self,i):
        ti,di = self.time_depth_indices(i)
        ds = self.ds.isel(time = ti,depth = di) 
        # ds = ds.isel(**{f"{k0}{k1}":slice(1000,1960) for k0 in 'u t'.split() for k1 in 'lat lon'.split()})
        return ds
    @property
    def ugrid(self,):
        ds = self.get_hres_dataset(0)
        ugrid,_ = get_grid_vars(ds)
        return ugrid
    @property
    def tgrid(self,):
        ds = self.get_hres_dataset(0)
        _,tgrid = get_grid_vars(ds)
        return tgrid
    @property
    def ugrid_subgrid_forcing(self,):
        if self._ugrid_subgrid_forcing is None:
            self._ugrid_subgrid_forcing = gcm_lsrp_subgrid_forcing(self.sigma,self.ugrid)
        return self._ugrid_subgrid_forcing
    
    @property
    def tgrid_subgrid_forcing(self,):
        if self._tgrid_subgrid_forcing is None:
            self._tgrid_subgrid_forcing =gcm_lsrp_subgrid_forcing(self.sigma,self.tgrid)
        return self._tgrid_subgrid_forcing

    @property
    def grid_interpolation(self,):
        if self._grid_interpolation is None:
            self._grid_interpolation = ugrid2tgrid_interpolation(self.ugrid,self.tgrid)
        return self._grid_interpolation

    def _base_get_hres(self,i,):
        ds = self.get_hres_dataset(i)
        u,v,temp = ds.u.load(),ds.v.load(),ds.temp.load()
        u = u.rename(ulat = "lat",ulon = "lon")
        v = v.rename(ulat = "lat",ulon = "lon")
        temp = temp.rename(tlat = "lat",tlon = "lon")
        u.name = 'u'
        v.name = 'v'
        temp.name = 'temp'
        return u,v,temp
   

    def get_mask(self,i):
        _,di = self.time_depth_indices(i)
        depthval = self.ds.depth.values[di]
        if self.wet_mask is None:
            self.build_mask(i)
        elif depthval not in self.wet_mask.depth.values:
            self.build_mask(i)
        return self.wet_mask.isel(depth = [di])

    def join_wet_mask(self,mask):
        def drop_time(mask):
            if 'time' in mask.dims:
                mask = mask.isel(time = 0)
            if 'time' in mask.coords:
                mask = mask.drop('time')
            return mask
        mask = drop_time(mask)
        if self.wet_mask is None:
            self.wet_mask = mask
        else:
            self.wet_mask = xr.merge([self.wet_mask,mask]).wet_mask
            

    def build_mask(self,i):
        u,v,temp = self._base_get_hres(i)
        fields = self.fields2forcings(i,u,v,temp,wet_masked = False)
        mask_ = None
        for val in fields.values():
            mask__ = xr.where(np.isnan(val),1,0)
            if mask_ is None:
                mask_ = mask__
            else:
                mask_ += mask__
        mask_.name = 'wet_mask'
        mask_ = xr.where(mask_>0,0,1)
        mask_ = mask_.isel(time = 0)
        self.join_wet_mask(mask_)
        return mask_
    def get_forcings(self,i):
        u,v,temp = self._base_get_hres(i)
        return self.fields2forcings(i,u,v,temp)
    def fields2forcings(self,i,u,v,temp):
        # if lsrp:
        #     def project(var,cg):
        #         cvar = cg(var)
        #         return cg(cvar,inverse = True)
        #     u = project(u,self.ugrid_subgrid_forcing.coarse_grain)
        #     v = project(v,self.ugrid_subgrid_forcing.coarse_grain)
        #     temp = project(temp,self.tgrid_subgrid_forcing.coarse_grain)

        u_t,v_t = self.grid_interpolation(u,v)
        uvars = dict(u=u,v=v)
        tvars = dict(u = u_t, v = v_t,temp = temp,)
        uvars = unbind(self.ugrid_subgrid_forcing(uvars,'u v'.split(),'Su Sv'.split()))
        tvars = unbind(self.tgrid_subgrid_forcing(tvars,'temp '.split(),'Stemp '.split()))
        def pass_gridvals(tgridvaldict,ugridvaldict):
            assert len(ugridvaldict) > 0
            ugridval = list(ugridvaldict.values())[0]
            for key,tgridval in tgridvaldict.items():
                for key_ in 'lat lon'.split():
                    tgridval[key_] = ugridval[key_]
                tgridvaldict[key] = tgridval
            return tgridvaldict
        tvars = pass_gridvals(tvars,uvars)
        fvars =  dict(uvars,**tvars)
        fvars = self.expand_dims(i,fvars,time = True,depth = True)
        return concat(**fvars)
    
        
    def expand_dims(self,i,fields,time = True,depth = True):
        ti,di = self.time_depth_indices(i)
        _time = self.ds.time.values[ti]
        _depth = self.ds.depth.values[di]
        dd = dict()
        if time:
            dd['time'] = [_time]
        if depth:
            dd['depth'] = [_depth]
        
        if isinstance(fields,dict):
            fields = {key:val.expand_dims(dd).compute() for key,val in fields.items()}
        else:
            fields = fields.expand_dims(dd)
        return fields
    # def get_forcings_with_mask(self,i,):
    #     fields = self.get_forcings(i,)
    #     fields = concat(**fields)
    #     return fields
    def __getitem__(self,i):
        ds = self.get_forcings(i,)
        # ds_lsrp = ds_lsrp.rename(**{f"S{key}":f"S{key}_res" for key in 'u v temp'.split()})
        # for key in 'u v temp'.split():
        #     ds_lsrp = ds_lsrp.drop(key)
        # ds = xr.merge([ds,ds_lsrp])
        return tonumpydict(ds)