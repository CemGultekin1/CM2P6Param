from typing import Callable
from transforms.coarse_grain_inversion import coarse_grain_inversion_weights
from utils.paths import coarse_graining_projection_weights_path
from utils.slurm import flushed_print
from utils.xarray import tonumpydict
import xarray as xr
from transforms.coarse_grain import coarse_graining_2d_generator
from transforms.grids import get_grid_vars, logitudinal_expansion, logitudinal_expansion_dataset, trim_expanded_longitude
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
    def coarse_graining_crop(self,):
        return 5

    def get_hres(self,i,fillna = False):
        di = i%len(self.ds.depth)
        ti = i//len(self.ds.depth)
        ds = self.ds.isel(time = ti,depth = di)
        # ds = ds.isel(ulon = slice(1000,1800),ulat = slice(1000,1600),tlon = slice(1000,1800),tlat = slice(1000,1600))
        
        if fillna:
            ds = ds.fillna(0)
        u,v,T = ds.u,ds.v,ds.T
        ugrid,tgrid = get_grid_vars(ds)
        u = u.rename(ulat = "lat",ulon = "lon")
        v = v.rename(ulat = "lat",ulon = "lon")
        T = T.rename(tlat = "lat",tlon = "lon")
        return ds.time.values,ds.depth.values,u.load(),v.load(),T.load(),ugrid,tgrid
    def get_gridvars(self,):
        return None,None
    def init_coarse_graining(self,i):
        time,depth,u,v,T,ugrid,tgrid=self.get_hres(i)

        u = logitudinal_expansion(u,self.coarse_graining_half_spread)
        v = logitudinal_expansion(v,self.coarse_graining_half_spread)
        T = logitudinal_expansion(T,self.coarse_graining_half_spread)

        ugrid = logitudinal_expansion_dataset(ugrid,self.coarse_graining_half_spread)
        tgrid = logitudinal_expansion_dataset(tgrid,self.coarse_graining_half_spread)

        cgu = coarse_graining_2d_generator(ugrid,self.sigma,wetmask = True)
        cgt = coarse_graining_2d_generator(tgrid,self.sigma,wetmask = True)


        dry_cgu = coarse_graining_2d_generator(ugrid,self.sigma,wetmask = False)
        dry_cgt = coarse_graining_2d_generator(tgrid,self.sigma,wetmask = False)

        self.coarse_grain =  (cgu,cgt)
        self.dry_coarse_grain =  (dry_cgu,dry_cgt)
        return time,depth,u,v,T,ugrid,tgrid
    def subgrid_forcing(self,u,v,T):
        sfds = subgrid_forcing(u,v,T,*self.coarse_grain)
        sfds = trim_expanded_longitude(sfds,expansion = self.coarse_graining_crop)
        return sfds
    def hres2lres(self,i):
        if not self.initiated:
            time,depth,u,v,T = self.init_coarse_graining(i)            
        else:
            time,depth,u,v,T = self.get_hres(i,)
            u = logitudinal_expansion(u,self.coarse_graining_half_spread)
            v = logitudinal_expansion(v,self.coarse_graining_half_spread)
            T = logitudinal_expansion(T,self.coarse_graining_half_spread)
        

        sfds = subgrid_forcing(u,v,T,*self.coarse_grain)
        sfds = trim_expanded_longitude(sfds,expansion = self.coarse_graining_crop)

        sfds = sfds.expand_dims(dim = {"time": [time]},axis=0)
        sfds = sfds.expand_dims(dim = {"depth": [depth]},axis=1)
        sfds.to_netcdf('forcings.nc')
        return sfds
    def __getitem__(self,i):
        ds = self.hres2lres(i)
        return tonumpydict(ds)




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
            time,depth,u,v,T,ugrid,tgrid = self.init_coarse_graining(i)
        else:
            flushed_print('time,depth,u,v,T,ugrid,tgrid = self.get_hres(i,)')
            time,depth,u,v,T,ugrid,tgrid = self.get_hres(i,)
            flushed_print('u = logitudinal_expansion(u,self.coarse_graining_half_spread)')
            u = logitudinal_expansion(u,self.coarse_graining_half_spread)
            v = logitudinal_expansion(v,self.coarse_graining_half_spread)
            T = logitudinal_expansion(T,self.coarse_graining_half_spread)
            flushed_print('ugrid = logitudinal_expansion_dataset(ugrid,self.coarse_graining_half_spread)')
            ugrid = logitudinal_expansion_dataset(ugrid,self.coarse_graining_half_spread)
            tgrid = logitudinal_expansion_dataset(tgrid,self.coarse_graining_half_spread)
        if not self.initiated:
            import numpy as np
            def replace_values(u):
                uv = u.values.copy()
                randentry = np.random.randn(*uv.shape)

                uv[uv==uv] = 0
                uv[uv!=uv] = randentry[uv!=uv]
                
                dims = u.dims
                coords = u.coords
                return xr.DataArray(
                    data = uv,
                    dims = dims,
                    coords = coords,
                    name = u.name
                )
            
            u1,v1,T1 = replace_values(u),replace_values(v),replace_values(T)

            sfds1 = subgrid_forcing(u1,v1,T1,ugrid,tgrid,*self.dry_coarse_grain)
            sfds1 = trim_expanded_longitude(sfds1,expansion = self.coarse_graining_crop)
            ugridmask = sfds1.u*0
            tgridmask = sfds1.T*0
            for name in list(sfds1.data_vars):
                vals = sfds1[name]
                print(name,np.any(np.isnan(vals.values)))
                vals = xr.where((np.abs(vals) + np.isnan(vals))>0,1,0)
                if 'u' in vals.dims[0]:
                    ugridmask = ugridmask + vals
                else:
                    tgridmask = tgridmask + vals
            ugridmask = xr.where( (ugridmask>0)  + np.isnan(ugridmask),1,0)
            tgridmask = xr.where( (tgridmask>0)  + np.isnan(tgridmask),1,0)
            ugridmask.name = 'ugrid_wetmask'
            tgridmask.name = 'tgrid_wetmask'
            wetmasks = xr.merge([ugridmask,tgridmask])
            self.wetmasks = wetmasks
            self.initiated = True

        sfds = subgrid_forcing(u,v,T,ugrid,tgrid,*self.coarse_grain)
        psfds = subgrid_forcing(u,v,T,ugrid,tgrid,*self.coarse_grain,projections = self.projections)
        psfds_vars = {f"lsrp_{key}":val for key,val in psfds.data_vars.items()}

        sfds = xr.Dataset(
            data_vars = dict(
                sfds.data_vars,**psfds_vars
            )
        )
        sfds = trim_expanded_longitude(sfds,expansion = self.coarse_graining_crop)
        
        sfds = sfds.expand_dims(dim = {"time": [str(time)]},axis=0)
        sfds = sfds.expand_dims(dim = {"depth": [depth]},axis=1)
        sfds = xr.merge([sfds,self.wetmasks])
        sfds.to_netcdf('subgrid_forcings.nc',mode = 'w')
        raise Exception
        return sfds

    def save_projections(self,):
        _,_,_,_,_,ugrid,tgrid = self.get_hres(0,fillna = True)

        ugrid = logitudinal_expansion_dataset(ugrid,self.coarse_graining_half_spread)
        tgrid = logitudinal_expansion_dataset(tgrid,self.coarse_graining_half_spread)
        
        projections = coarse_grain_inversion_weights(ugrid,tgrid,self.sigma)
        print(coarse_graining_projection_weights_path(self.sigma))
        print(projections)
        projections.to_netcdf(coarse_graining_projection_weights_path(self.sigma),mode = 'w')