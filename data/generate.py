from typing import Callable
from transforms.coarse_grain_inversion import coarse_grain_inversion_weights
from utils.paths import coarse_graining_projection_weights_path
import xarray as xr
from transforms.coarse_grain import coarse_graining_2d_generator
from transforms.grids import logitudinal_expansion, trim_expanded_longitude
from transforms.subgrid_forcing import subgrid_forcing

class HighResCm2p6:
    ds : xr.Dataset
    sigma : int
    half_spread : int
    coarse_grain : Callable
    initiated : bool
    def __init__(self,ds:xr.Dataset,sigma,*args,**kwargs):
        self.ds = ds.copy()
        self.sigma = sigma
        self.coarse_grain = None
        self.initiated = False
    @property
    def depth(self,):
        return self.ds.depth
    def __len__(self,):
        return len(self.ds.time)
    @property
    def coarse_graining_half_spread(self,):
        return int(self.sigma*6)
    
    @property
    @staticmethod
    def coarse_graining_crop():
        return 5

    def get_hres(self,i,fillna = False):
        di = i%len(self.ds.depth)
        ti = i//len(self.ds.depth)
        ds = self.ds.isel(time = ti,depth = di)
        if fillna:
            ds = ds.fillna(0)
        u,v,T = ds.u,ds.v,ds.T
        u = u.rename(ulat = "lat",ulon = "lon")
        v = v.rename(ulat = "lat",ulon = "lon")
        T = T.rename(tlat = "lat",tlon = "lon")
        return u.load(),v.load(),T.load()
    def init_coarse_graining(self,i):
        u,v,T=self.get_hres(i)

        u = logitudinal_expansion(u,self.coarse_graining_half_spread)
        v = logitudinal_expansion(v,self.coarse_graining_half_spread)
        T = logitudinal_expansion(T,self.coarse_graining_half_spread)
     
        cgu = coarse_graining_2d_generator(u,self.sigma,wetmask = True)
        cgt = coarse_graining_2d_generator(T,self.sigma,wetmask = True)

        self.coarse_grain =  (cgu,cgt)
        return u,v,T
    def hres2lres(self,i):
        if not self.initiated:
            u,v,T = self.init_coarse_graining(i)
            self.initiated = True
        else:
            u,v,T = self.get_hres(i,)
            u = logitudinal_expansion(u,self.coarse_graining_half_spread)
            v = logitudinal_expansion(v,self.coarse_graining_half_spread)
            T = logitudinal_expansion(T,self.coarse_graining_half_spread)
        sfds = subgrid_forcing(u,v,T,*self.coarse_grain)
        sfds = trim_expanded_longitude(sfds,expansion = self.coarse_graining_crop)
        sfds = sfds.expand_dims(dim = {"time": [self.ds.time.values[i]]},axis=0)
        sfds = sfds.expand_dims(dim = {"depth": [self.depth]},axis=1)
        return sfds
    def __getitem__(self,i):
        return self.hres2lres(i)




class ProjectedHighResCm2p6(HighResCm2p6):
    def __init__(self, ds: xr.Dataset, sigma, *args, **kwargs):
        super().__init__(ds, sigma, *args, **kwargs)
        self.projections = None
    
    def init_coarse_graining(self, i):
        path = coarse_graining_projection_weights_path(self.sigma)
        projections = xr.open_dataset(path)
        self.projections = projections.load()
        return super().init_coarse_graining(i)
    def hres2lres(self,i):
        if not self.initiated:
            u,v,T = self.init_coarse_graining(i)
            self.initiated = True
        else:
            u,v,T = self.get_hres(i,)
            u = logitudinal_expansion(u,self.coarse_graining_half_spread)
            v = logitudinal_expansion(v,self.coarse_graining_half_spread)
            T = logitudinal_expansion(T,self.coarse_graining_half_spread)
        sfds = subgrid_forcing(u,v,T,*self.coarse_grain)
        psfds = subgrid_forcing(u,v,T,*self.coarse_grain,projections = self.projections)
        psfds_vars = {f"lsrp_{key}":val for key,val in psfds.data_vars.items()}

        sfds = xr.Dataset(
            data_vars = dict(
                sfds.data_vars,**psfds_vars
            )
        )
        sfds = trim_expanded_longitude(sfds,expansion = self.coarse_graining_crop)
        sfds = sfds.expand_dims(dim = {"time": [self.ds.time.values[i]]},axis=0)
        sfds = sfds.expand_dims(dim = {"depth": [self.depth]},axis=1)
        return sfds

    def save_projections(self,):
        ti = 0
        di = 0
        ds = self.ds.isel(time = ti,depth = di)
        ds = ds.fillna(0)
        u,v,T = ds.u,ds.v,ds.T
        u,v,T = u.load(),v.load(),T.load()
        
        T = logitudinal_expansion(T,self.coarse_graining_half_spread,prefix='t')
        u = logitudinal_expansion(u,self.coarse_graining_half_spread,prefix='u')
        data_vars = {
            'u':u,
            'T':T
        }
        utgrid = xr.Dataset(data_vars)
        projections = coarse_grain_inversion_weights(utgrid,self.sigma)
        print(coarse_graining_projection_weights_path(self.sigma))
        projections.to_netcdf(coarse_graining_projection_weights_path(self.sigma),mode = 'w')