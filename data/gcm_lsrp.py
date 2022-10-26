import itertools
from typing import Dict, Tuple
from data.vars import LSRP_NAMES
from transforms.lsrp import get_projection_weights
from utils.xarray import no_nan_input_mask
import xarray as xr
from data.gcm_forcing import CM2p6Dataset, SingleDomain
import numpy as np
from models.nets.lsrp import ConvolutionalLSRP
from transforms.grids import divide2equals
from transforms.subgrid_forcing import subgrid_forcing
class LSRSingleDomain(SingleDomain):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        proj_weights = get_projection_weights(self.sigma)
        self.proj_weights = (proj_weights.proj_lat.values,proj_weights.proj_lon.values)
        self.lsrp_forcing_names = LSRP_NAMES.copy()

    def hres_projection(self,*args):
        args = list(args)
        latp,lonp = self.proj_weights
        for i,arg in enumerate(args):
            ds = args[i].copy()
            ds.data = (latp.T@(latp@arg.values@lonp.T))@lonp
            args[i] = ds
        return tuple(args)
    def projected_hres(self,*args,fillna = False):
        u,v,T = self.get_hres(*args,fillna= False)
        u0,v0,T0 = u.fillna(0),v.fillna(0),T.fillna(0)
        up0,vp0,Tp0 = self.hres_projection(u0,v0,T0)
        if not fillna:
            up = xr.where(u == np.nan, np.nan, up0)
            vp = xr.where(v == np.nan, np.nan, vp0)
            Tp = xr.where(T == np.nan, np.nan, Tp0)
            return u,v,T,up,vp,Tp
        else:
            return u0,v0,T0,up0,vp0,Tp0
    def hres2lres(self, i):
        if self.coarse_grain is None:
            self.init_coarse_graining()
        if self.all_land:
            Zs = self.nan_map
            return dict(fields = Zs, forcings = Zs,lsrp_res_forcings = Zs)
        u,v,T,up,vp,Tp = self.projected_hres(i,fillna = True)
        u,v,T = self.fix_grid(u,Tgrid= False), self.fix_grid(v,Tgrid= False), self.fix_grid(T,Tgrid= True)
        up,vp,Tp  = self.fix_grid(up,Tgrid= False), self.fix_grid(vp,Tgrid= False), self.fix_grid(Tp,Tgrid= True)

        F,U = subgrid_forcing(u,v,T,*self.coarse_grain)
        Fp,_ = subgrid_forcing(up,vp,Tp,*self.coarse_grain)
        RF = F- Fp
        F,U,RF = (A.sel(**self.final_boundaries) for A in (F,U,RF))
        M0 = self.wet_mask.sel(**self.final_boundaries) == 1
        RF = xr.where(M0, RF, np.nan)
        F = xr.where(M0, F, np.nan)
        U = xr.where(M0, U, np.nan)
        K =  dict(fields = U,forcings = F,lsrp_res_forcings = RF)
        return K
    def lres2lres(self,i):
        UF = super().lres2lres(i)
        U,F = UF.pop('fields'),UF.pop('forcings')
        RFlist = [F[lsrp_force] for lsrp_force in self.lsrp_forcing_names]
        lfn = self.lsrp_forcing_names[0]
        RF = RFlist[0].to_dataset(name = lfn)
        for i,lfn in enumerate(self.lsrp_forcing_names[1:]):
            RF[lfn] = RFlist[i+1]
        return dict(fields = U,forcings = F, lsrp_res = RF)

class ConvLSRSingleDomain(SingleDomain):
    lsrp_model: ConvolutionalLSRP
    lsrp_mask : xr.DataArray
    def __init__(self,ds:xr.Dataset,sigma,*args,boundaries = None,lsrp_span = 5,half_spread = 0,**kwargs):
        self.lsrp_span = lsrp_span
        half_spread += lsrp_span*sigma
        super().__init__(ds,sigma,*args,boundaries = boundaries,half_spread = half_spread,**kwargs)
        lat,lon = self.final_local_lres_coords
        spl = slice(lsrp_span,-lsrp_span)
        lat = lat[spl]
        lon = lon[spl]
        self.final_boundaries = {"lat": slice(*lat[[0,-1]]),"lon":slice(*lon[[0,-1]])}
        self.final_local_lres_coords = lat,lon
        self.lsrp_mask = None
        self.lsrp_model = None
    def set_half_spread(self,addsp):
        addsp += self.lsrp_span*self.sigma
        super().set_half_spread(addsp)
        lat,lon = self.final_local_lres_coords
        spl = slice(self.lsrp_span,-self.lsrp_span)
        lat = lat[spl]
        lon = lon[spl]
        self.final_boundaries = {"lat": slice(*lat[[0,-1]]),"lon":slice(*lon[[0,-1]])}
        self.final_local_lres_coords = lat,lon
        self.lsrp_mask = None
        self.lsrp_model = None


    def build_lsrp_mask(self,):
        mask = 1 - no_nan_input_mask(self.wet_mask,self.lsrp_span, lambda x: x==0)
        return mask

    def __getitem__(self, i):
        if self.coarse_grain is None:
            self.init_coarse_graining()
            clat,_ = self.local_lres_coords
            self.lsrp_model = ConvolutionalLSRP(self.sigma,clat,self.lsrp_span)
            self.lsrp_mask = self.build_lsrp_mask()
        if self.all_land:
            Zs = self.nan_map
            return dict(fields = Zs, forcings = Zs,lsrp_res_forcings = Zs)
        u,v,T = self.get_hres(i,fillna = True)
        F,U = subgrid_forcing(u,v,T,*self.coarse_grain)
        RF = self.lsrp_model.forward(**U)
        F,U,RF = (A.sel(**self.final_boundaries) for A in (F,U,RF))
        M0 = self.wet_mask.sel(**self.final_boundaries) == 1
        RF = xr.where(M0, RF, np.nan)
        F = xr.where(M0, F, np.nan)
        U = xr.where(M0, U, np.nan)
        K =  dict(fields = U,forcings = F,lsrp_res_forcings = F - RF)
        return K


class DividedDomain(CM2p6Dataset):
    cgs : Dict[Tuple[int,int],SingleDomain]
    def __init__(self,*args,parts = (1,1),**kwargs):
        super().__init__(*args,**kwargs)
        if self.coarse_grain_needed:
            lat,lon,_,_ = self.global_hres_coords
        else:
            lat,lon= self.global_lres_coords
        bds = divide2equals(lat,lon,parts[0],parts[1],*self.preboundaries)
        self.parts = parts
        self.cgs = {}
        self.linsupres = kwargs.pop("linsupres",False)
        kwargs.pop('boundaries')
        if not self.linsupres or not self.coarse_grain_needed:
            constructor = SingleDomain
        else:
            constructor = LSRSingleDomain
        for i,j in self.iterate_over_parts():
            self.cgs[(i,j)] = constructor(self.ds,self.sigma,boundaries=bds[(i,j)],**kwargs)
    def set_time_constraint(self, t0, t1):
        super().set_time_constraint(t0, t1)
        for i,j in self.iterate_over_parts():
            self.cgs[(i,j)].set_time_constraint(t0,t1)
    @property
    def ntime(self,):
        return super().__len__()
    def set_half_spread(self,spread):
        for i,j in self.iterate_over_parts():
            self.cgs[(i,j)].set_half_spread(spread)
    def post__getitem__(self,i,j,t)-> Dict[str,xr.Dataset]:
        return self.cgs[(i,j)][t]
    def iterate_over_parts(self,):
        for i,j in itertools.product(range(self.parts[0]),range(self.parts[1])):
            yield i,j
    @property
    def shapes(self,):
        shapes = np.empty((self.parts[0],self.parts[1],2),dtype = int)
        for i,j, in self.iterate_over_parts():
            shapes[i,j,:] = self.cgs[(i,j)].shape
        return shapes
    def get_max_shape(self,):
        lat,lon = 0,0
        shp = self.shapes
        for i,j in self.iterate_over_parts():
            lat_,lon_ = shp[i,j,:]
            lat = np.maximum(lat_,lat)
            lon = np.maximum(lon_,lon)
        return lat,lon
    def factor_index(self,i):
        lat,lon = self.parts
        li = i%lat
        i = i//lat
        lj = i%lon
        i = i//lon
        t = i%self.ntime
        return li,lj,t

    def __getitem__(self,i):
        li,lj,t = self.factor_index(i)
        return dict(self.post__getitem__(li,lj,t),**dict(ilat = li,ilon = lj, itime = t,depth = self.depth))
    def __len__(self,):
        lon,lat = self.parts
        return super().__len__()*lon*lat
