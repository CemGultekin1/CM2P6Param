import copy
import itertools
from typing import Dict, Tuple
from utils.xarray import no_nan_input_mask #concat, 
import xarray as xr
import numpy as np
from transforms.grids import bound_grid, divide2equals, fix_grid, larger_longitude_grid, lose_tgrid


class CM2p6Dataset:
    ds : xr.Dataset
    sigma : int
    half_spread : int
    preboundaries : Tuple[int,...]
    global_hres_coords : Tuple[np.ndarray,...] #3 periods of longitude
    global_lres_coords: Tuple[np.ndarray,...] #3 periods of longitude
    def __init__(self,ds:xr.Dataset,sigma,*args,boundaries = None,half_spread = 0,**kwargs):
        self.ds = ds.copy()
        self.sigma = sigma
        self.half_spread = half_spread
        self.global_hres_coords = [None]*4
        def flatten_tuple_list(l_):
            # varnames = list(ds.data_vars)
            l = []
            for n in l_:
                # if n in varnames:
                l.append(n)
            return l
        # if boundaries is None:
        self.requested_boundaries = None if isinstance(boundaries,tuple) else boundaries
    
        varnames = kwargs.get('var_grouping',None)
        if varnames is not None:
            self.field_names = flatten_tuple_list(varnames[0])
            self.forcing_names = flatten_tuple_list(varnames[1])
        else:
            self.field_names = None
            self.forcing_names  = None
        
        self.global_lres_coords = self.ds.ulat.values,larger_longitude_grid(self.ds.ulon.values)
        self.preboundaries = (-90,90,-180,180)
        

    @property
    def depth(self,):
        return self.ds.depth.values
    def set_time_constraint(self,t0,t1):
        x = copy.deepcopy(self)
        nt = len(self.ds.time)
        t0,t1 = np.floor(nt*t0),np.ceil(nt*t1)
        t0 = int(np.maximum(t0,0))
        t1 = int(np.minimum(t1,len(self.ds.time)))
        x.ds = self.ds.isel(time = slice(t0,t1))
        return x
    def ntimes(self,):
        return len(self.ds.time)
    def __len__(self,):
        return len(self.ds.time)

    @property
    def lres_spread(self,):
        return self.half_spread

    def locate(self,*args,lat = True,):
        clat,clon = self.global_lres_coords
        if lat:
            cc = clat
        else:
            cc = clon
        locs = []
        for lat in args:
            locs.append(np.argmin(np.abs(cc - lat)))
        return dict(locs = locs,len = len(cc))

   
class SingleDomain(CM2p6Dataset):
    local_lres_coords :Tuple[np.ndarray,...] # larger lres grid
    final_local_lres_coords : Tuple[np.ndarray,...] # smaller lres grid
    initiated : bool
    all_land : bool
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.confine(*self.preboundaries)
        self.initiated = False
        self.all_land = None
        self._wetmask = None
        self._forcingmask = None
        

    @property
    def shape(self,):
        clat,clon = self.final_local_lres_coords
        return len(clat), len(clon)
    def set_half_spread(self,addsp):
        self.half_spread = addsp
        self.confine(*self.preboundaries)
        if self.initiated and self.coarse_grain_needed:
            self.coarse_grain = self.init_coarse_graining()

    def confine(self,latmin,latmax,lonmin,lonmax):
        ulat,ulon = self.global_lres_coords
        bulat,bulon,lat0,lat1,lon0,lon1 = bound_grid(ulat,ulon,latmin,latmax,lonmin,lonmax,self.lres_spread)
        self.boundaries = {"lat":slice(lat0,lat1),"lon": slice(lon0,lon1)}
        self.final_boundaries = {"lat":slice(lat0,lat1),"lon": slice(lon0,lon1)}
        self.local_lres_coords = bulat,bulon
        self.final_local_lres_coords = bulat,bulon
        
    def fix_grid(self,u, ):
        latlon = self.local_lres_coords
        dims = u.dims
        if 'lat' in dims and 'lon' in dims:
            return fix_grid(u,latlon)#at_name = "ulat",lon_name = "ulon")
        else:
            return u
    @property
    def fieldwetmask(self,):
        if self._wetmask is None:
            ds = self.ds.isel(time =0).load()
            ds = lose_tgrid(ds)
            ds = self.get_grid_fixed_lres(ds)
            wetmask = ds.ugrid_wetmask + ds.tgrid_wetmask
            wetmask = xr.where(wetmask > 0,1,0)
            if self.requested_boundaries is not None:
                wmask = wetmask.values
                bmask = wmask*0 + 1
                lat = wetmask.lat.values 
                lon = wetmask.lon.values
                for lat0,lat1,lon0,lon1 in self.requested_boundaries:
                    latmask = (lat >= lat0)*(lat <= lat1)
                    lonmask = (lon >= lon0)*(lon <= lon1)
                    mask = latmask.reshape([-1,1])@lonmask.reshape([1,-1])
                    bmask = mask  + bmask
                bmask = (bmask > 1).astype(float)
                wmask = wmask*bmask
                wetmask = xr.DataArray(
                    data = wmask,
                    dims = ['lat','lon'],
                    coords = {'lat':lat,'lon':lon},
                    name = 'wetmask'
                )
            self._wetmask = wetmask
            # import matplotlib.pyplot as plt
            # self._wetmask.plot()
            # plt.savefig('wetmask.png')
            # plt.close()
        return self._wetmask
    @property
    def forcingwetmask(self,):
        if self._forcingmask is None:
            forcing_mask = no_nan_input_mask(self._wetmask,self.half_spread,lambda x: x==0,same_size = True)
            self._forcingmask =  xr.where(forcing_mask==0,1,0)
        return self._forcingmask
            
    def get_dataset(self,t):
        ds = self.ds.isel(time =t).load()
        # print(ds)
        ds = lose_tgrid(ds)
        ds = self.get_grid_fixed_lres(ds)
        for prefix in '0 1'.split():
            lsrp_vars = [v for v in list(ds.data_vars) if prefix in v]
            for lv in lsrp_vars:
                lsrp_forcing = ds[lv].values
                forcing = ds[lv.replace(prefix,'')].values
                if 'tr_depth' in ds[lv].dims:
                    forcing = np.stack([forcing],axis = 0)
                lsrp_res_forcing = forcing - lsrp_forcing
                ds[lv.replace(prefix,f'{prefix}_res')] = (ds[lv].dims,lsrp_res_forcing)
        


        def apply_mask(ds,wetmaskv,keys):
            for name in keys:
                v = ds[name].values
                vshp = list(v.shape)
                v = v.reshape([-1] + vshp[-2:])
                v[:,wetmaskv<1] = np.nan
                v = v.reshape(vshp)
                ds[name] = (ds[name].dims,v)
            return ds

        ds = ds.drop('tgrid_wetmask').drop('ugrid_wetmask')
        # forcing_mask = no_nan_input_mask(wetmask,self.half_spread,lambda x: x==0,same_size = True)
        # forcing_mask = xr.where(forcing_mask==0,1,0)
        ds = apply_mask(ds,self.fieldwetmask.values,list(ds.data_vars))
        ds = apply_mask(ds,self.forcingwetmask.values,[field for field in list(ds.data_vars) if 'S' in field])
        # ds['time'] = self.ds.time.values[t]
        # ds['depth'] = self.ds.depth.values

        return ds

    def get_grid_fixed_lres(self,ds):
        fields = list(ds.data_vars.keys())
        var = []
        for field in fields:
            v = self.fix_grid(ds[field])
            v.name = field
            var.append(v)
        U = xr.merge(var)
        return  U

    def __getitem__(self,i):
        ds = self.get_dataset(i)
        # print(ds)
        return ds
