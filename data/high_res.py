from typing import Callable
from utils.no_torch_xarray import concat, tonumpydict
import xarray as xr
from transforms.grids import get_grid_vars, ugrid2tgrid_interpolation
from transforms.subgrid_forcing import base_lsrp_subgrid_forcing, gcm_lsrp_subgrid_forcing, scipy_subgrid_forcing,greedy_scipy_lsrp_subgrid_forcing
import numpy as np

class HighResCm2p6:
    ds : xr.Dataset
    sigma : int
    half_spread : int
    coarse_grain : Callable
    initiated : bool
    def __init__(self,ds:xr.Dataset,sigma,*args,section = [0,1],**kwargs):
        self.ds = ds.copy()#.isel({f"{prefix}{direction}":slice(1500,1800) for prefix in 'u t'.split() for direction in 'lat lon'.split()})
        self.sigma = sigma
        self.initiated = False
        self.wet_mask = None
        self._ugrid_subgrid_forcing = None
        self._tgrid_subgrid_forcing = None
        self._ugrid_scipy_forcing = None
        self._tgrid_scipy_forcing = None
        self._grid_interpolation = None
        self._scipy_forcing_class = scipy_subgrid_forcing
        if kwargs.get('filtering') == 'gcm':
            self.forcing_class = gcm_lsrp_subgrid_forcing
        else:
            assert kwargs.get('filtering') == 'gaussian'
            self.forcing_class = greedy_scipy_lsrp_subgrid_forcing
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
    def ugrid_scipy_forcing(self,)->scipy_subgrid_forcing:
        if self._ugrid_scipy_forcing is None:
            self._ugrid_scipy_forcing :scipy_subgrid_forcing= scipy_subgrid_forcing(self.sigma,self.ugrid)
        return self._ugrid_scipy_forcing

    @property
    def tgrid_scipy_forcing(self,)->scipy_subgrid_forcing:
        if self._tgrid_scipy_forcing is None:
            self._tgrid_scipy_forcing :scipy_subgrid_forcing= scipy_subgrid_forcing(self.sigma,self.tgrid)
        return self._tgrid_scipy_forcing
        
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
    def ugrid_subgrid_forcing(self,)-> base_lsrp_subgrid_forcing:
        if self._ugrid_subgrid_forcing is None:
            self._ugrid_subgrid_forcing : base_lsrp_subgrid_forcing = self.forcing_class(self.sigma,self.ugrid)
        return self._ugrid_subgrid_forcing
    
    @property
    def tgrid_subgrid_forcing(self,)-> base_lsrp_subgrid_forcing:
        if self._tgrid_subgrid_forcing is None:
            self._tgrid_subgrid_forcing :base_lsrp_subgrid_forcing =self.forcing_class(self.sigma,self.tgrid)
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
        return self.wet_mask.sel(depth = [depthval])

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
            self.wet_mask = xr.merge([self.wet_mask,mask])#.wet_mask
    def build_mask(self,i):
        u,v,temp = self._base_get_hres(i)
        fields = self.fields2forcings(i,u,v,temp,scipy_filtering = True)
        mask_ = None
        for val in fields.values():
            mask__ = xr.where(np.isnan(val),1,0)
            if mask_ is None:
                mask_ = mask__
            else:
                mask_ += mask__
        mask_.name = 'interior_wet_mask'
        mask_ = xr.where(mask_>0,0,1)
        mask_ = mask_.isel(time = 0)
        self.join_wet_mask(mask_)
        return mask_
    def get_forcings(self,i):
        u,v,temp = self._base_get_hres(i)
        return self.fields2forcings(i,u,v,temp)
    def fields2forcings(self,i,u,v,temp,scipy_filtering = False):
        u_t,v_t = self.grid_interpolation(u,v)
        uvars = dict(u=u,v=v)
        tvars = dict(u = u_t, v = v_t,temp = temp,)
        def switch_grid_on_dictionary(ulres):
            ulres['u'],ulres['v'] = self.grid_interpolation(ulres['u'],ulres['v'])
        if scipy_filtering:
            uforcings,(uclres,ulres) = self.ugrid_scipy_forcing(uvars,'u v'.split(),'Su Sv'.split())
            switch_grid_on_dictionary(ulres)
            tforcings,(tcres,_) = self.tgrid_scipy_forcing(tvars,'temp '.split(),'Stemp '.split(),lres = ulres)
        else:
            uforcings,(uclres,ulres),(_,ulres0,uhres0) = self.ugrid_subgrid_forcing(uvars,'u v'.split(),'Su Sv'.split())
            switch_grid_on_dictionary(ulres)
            switch_grid_on_dictionary(ulres0)
            switch_grid_on_dictionary(uhres0)
            tforcings,(tcres,_),_ = self.tgrid_subgrid_forcing(tvars,'temp '.split(),'Stemp '.split(),lres = ulres,hres0 = uhres0,lres0 = ulres0,)
        uvars = dict(uforcings,**uclres)
        tvars = dict(tforcings,**tcres)
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
    def append_mask(self,ds,i):
        wetmask = self.get_mask(i)
        ds = xr.merge([ds,wetmask])
        # ds = xr.merge([wetmask])
        # ti,_ = self.time_depth_indices(i)
        # ds = ds.expand_dims(dim = {"time":[ti]},axis = 0)
        return ds
    def __getitem__(self,i):
        ds = self.get_forcings(i,)
        ds = self.append_mask(ds,i)
        return tonumpydict(ds)