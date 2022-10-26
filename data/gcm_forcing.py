import itertools
from typing import Callable, Tuple
from utils.xarray import concat, no_nan_input_mask
import xarray as xr
import numpy as np

from transforms.coarse_grain import coarse_graining_2d_generator


from transforms.grids import bound_grid, fix_grid, larger_longitude_grid, lose_tgrid, make_divisible_by_grid, trim_grid_nan_boundaries, ugrid2tgrid
from transforms.subgrid_forcing import subgrid_forcing


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
        self.wetmask_flag = kwargs.get('wetmask',False)
        def flatten_tuple_list(l_):
            # varnames = list(ds.data_vars)
            l = []
            for n in l_:
                # if n in varnames:
                l.append(n)
            return l
        if boundaries is None:
            boundaries = (-90,90,-180,180)
    
        varnames = kwargs.get('var_grouping')
        self.field_names = flatten_tuple_list(varnames[0])
        self.forcing_names = flatten_tuple_list(varnames[1])
        
        self.global_lres_coords = self.ds.ulat.values,larger_longitude_grid(self.ds.ulon.values)
        self.preboundaries = boundaries

    @property
    def depth(self,):
        return self.ds.depth.values
    def set_time_constraint(self,t0,t1):
        nt = len(self.ds.time)
        t0,t1 = np.floor(nt*t0),np.ceil(nt*t1)
        t0 = int(np.maximum(t0,0))
        t1 = int(np.minimum(t1,len(self.ds.time)))
        self.ds = self.ds.isel(time = slice(t0,t1))
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
    wet_mask : xr.DataArray
    initiated : bool
    all_land : bool
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.confine(*self.preboundaries)
        self.initiated = False
        self.all_land = None
        

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
        return fix_grid(u,latlon)#at_name = "ulat",lon_name = "ulon")

    def get_dataset(self,t):
        ds = self.ds.isel(time =t)
        ds = lose_tgrid(ds)
        lsrp_vars = [v for v in list(ds.data_vars) if 'lsrp' in v]
        for lv in lsrp_vars:
            lsrp_forcing = ds[lv].values
            forcing = ds[lv.replace('lsrp_','')]
            lsrp_res_forcing = forcing.values - lsrp_forcing
            ds[lv.replace('lsrp_','lsrp_res_')] = (forcing.dims,lsrp_res_forcing)
        return ds
    def get_grid_fixed_lres(self,i,fields):
        ds = self.get_dataset(i)
        print(fields,list(ds.data_vars))
        U = concat(**{field : self.fix_grid(ds[field]) for field in fields})
        return  U

    def init_coarse_masks(self,):
        ds = self.get_dataset(0)

        def get_mask(name):
            u = ds[name]
            u = self.fix_grid(u)
            return xr.where(no_nan_input_mask(u,0) == 0 ,1,0)
        
        cmask = None
        for key in "u v T".split():
            mask = get_mask(key)
            if cmask is None:
                cmask = mask
            else:
                cmask = cmask + mask
        self.wet_mask =  xr.where(cmask == 0 ,0,1)
        forcing_mask = no_nan_input_mask(self.wet_mask,self.half_spread,lambda x: x==1,same_size = True)
        self.forcing_mask = xr.where(forcing_mask>0,1,0)
        self.forcing_mask =  xr.where(self.forcing_mask == 0, 1,0)
        self.wet_mask = xr.where(self.wet_mask==0,1,0)
        self.all_land = np.mean(self.forcing_mask.values) > 1 - 1e-2

    @property
    def nan_map(self,):
        lat,lon = self.final_local_lres_coords
        u = [None]*len(self.field_names)
        for i,key in enumerate(self.field_names):
            u[i]  = xr.DataArray(data =np.ones(self.shape)*np.nan, dims = ["lat","lon"], coords = dict( lat =lat, lon= lon ),name = key)
        U = xr.merge(u)

        f = [None]*len(self.forcing_names)
        for i,key in enumerate(self.forcing_names):
            f[i]  = xr.DataArray(data =np.ones(self.shape)*np.nan, dims = ["lat","lon"], coords = dict( lat =lat, lon= lon ),name = key)
        F = xr.merge(f)
        return U,F

    def __getitem__(self,i):
        if not self.initiated:
            self.init_coarse_masks()
            self.initiated = True
        if self.all_land:
            Zs = self.nan_map
            return dict(fields = Zs, forcings = Zs)

        U = self.get_grid_fixed_lres(i, self.field_names)
        F = self.get_grid_fixed_lres(i, self.forcing_names)
        M1 = self.forcing_mask.sel(**self.final_boundaries) == 1
        M0 = self.wet_mask.sel(**self.final_boundaries) == 1
        F = xr.where(M1, F.sel(**self.final_boundaries), np.nan)
        U = xr.where(M0, U.sel(**self.final_boundaries), np.nan)
        return dict(fields = U,forcings = F)

