import itertools
from typing import Callable, Tuple
from utils.xarray import concat, no_nan_input_mask
import xarray as xr
import numpy as np

from transforms.coarse_grain import coarse_graining_2d_generator


from transforms.grids import bound_grid, fix_grid, larger_longitude_grid, make_divisible_by_grid, trim_grid_nan_boundaries, ugrid2tgrid
from transforms.subgrid_forcing import subgrid_forcing


class CM2p6Dataset:
    ds : xr.Dataset
    sigma : int
    half_spread : int
    preboundaries : Tuple[int,...]
    global_hres_coords : Tuple[np.ndarray,...] #3 periods of longitude
    global_lres_coords: Tuple[np.ndarray,...] #3 periods of longitude
    def __init__(self,ds:xr.Dataset,sigma,*args,boundaries = None,half_spread = 0,coarse_grain_needed = False,**kwargs):
        self.ds = ds.copy()
        self.coarse_grain_needed = coarse_grain_needed
        self.sigma = sigma
        self.half_spread = half_spread
        self.global_hres_coords = [None]*4
        self.wetmask_flag = kwargs.get('wetmask',False)
        def flatten_tuple_list(l_):
            varnames = list(ds.data_vars)
            l = []
            for n in l_:
                if n in varnames:
                    l.append(n)
            return l
        if boundaries is None:
            boundaries = (-90,90,-180,180)
        if coarse_grain_needed:
            if isinstance(boundaries,list):
                boundaries_ = []
                for bdr in boundaries:
                    boundaries_.append(make_divisible_by_grid(ds.rename({"ulon":"lon","ulat":"lat"}),sigma,*bdr))
                boundaries = boundaries_
            else:
                boundaries = make_divisible_by_grid(ds.rename({"ulon":"lon","ulat":"lat"}),sigma,*boundaries)
            ulat,ulon = ds.ulat.values,ds.ulon.values
            tlat,tlon = ugrid2tgrid(ulat,ulon)

            self.ds = self.ds.assign(tlat = tlat,tlon = tlon)
            ulon,tlon = larger_longitude_grid(ulon),larger_longitude_grid(tlon)
            self.global_hres_coords = ulat,ulon,tlat,tlon
            def coarsen(ulon):
                return  xr.DataArray(data = ulon).coarsen(dim_0 = sigma,boundary = "trim").mean().values
            self.global_lres_coords = coarsen(ulon),coarsen(ulat)
        else:
            varnames = kwargs.get('var_grouping')
            self.field_names = flatten_tuple_list(varnames[0])
            self.forcing_names = flatten_tuple_list(varnames[1])
            self.global_lres_coords = self.ds.lat.values,larger_longitude_grid(self.ds.lon.values)
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
    def hres_spread(self,):
        return self.coarse_graining_spread + self.half_spread*self.sigma

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

    @property
    def coarse_graining_spread(self,):
        if not self.coarse_grain_needed:
            return 0
        if self.wetmask_flag:
            return 0
        return self.sigma*3


class SingleDomain(CM2p6Dataset):
    coarse_grain : Callable
    local_hres_coords: Tuple[np.ndarray,...] # larger hres grid
    local_lres_coords :Tuple[np.ndarray,...] # larger lres grid
    final_local_lres_coords : Tuple[np.ndarray,...] # smaller lres grid
    wet_mask : xr.DataArray
    initiated : bool
    all_land : bool
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.confine(*self.preboundaries)
        self.coarse_grain = None
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
        if not self.coarse_grain_needed:
            ulat,ulon = self.global_lres_coords
            bulat,bulon,lat0,lat1,lon0,lon1 = bound_grid(ulat,ulon,latmin,latmax,lonmin,lonmax,self.lres_spread)
            self.boundaries = {"lat":slice(lat0,lat1),"lon": slice(lon0,lon1)}
            self.final_boundaries = {"lat":slice(lat0,lat1),"lon": slice(lon0,lon1)}
            self.local_lres_coords = bulat,bulon
            self.final_local_lres_coords = bulat,bulon
        else:
            ulat,ulon,tlat,tlon = self.global_hres_coords
            bulat,bulon,lat0,lat1,lon0,lon1 = bound_grid(ulat,ulon,latmin,latmax,lonmin,lonmax,self.hres_spread)
            self.boundaries = {"lat":slice(lat0,lat1),"lon": slice(lon0,lon1)}
            btlat,btlon,_,_,_,_ = bound_grid(tlat,tlon,latmin,latmax,lonmin,lonmax,self.hres_spread)
            self.local_hres_coords = bulat,bulon,btlat,btlon
            def coarsen_coord(ulat):
                return xr.DataArray(data=ulat,dims = ["x"],coords = dict(x = ulat)).coarsen(x = self.sigma,boundary = "trim").mean().x.values
            cbulat = coarsen_coord(bulat)
            cbulon = coarsen_coord(bulon)
            self.local_lres_coords = cbulat,cbulon
            cbulat,cbulon,lat0,lat1,lon0,lon1 = bound_grid(cbulat,cbulon,latmin,latmax,lonmin,lonmax,self.lres_spread)
            self.final_boundaries = {"lat":slice(lat0,lat1),"lon": slice(lon0,lon1)}
            self.final_local_lres_coords = cbulat,cbulon

    def fix_grid(self,u, Tgrid = False):
        if not self.coarse_grain_needed:
            latlon = self.local_lres_coords
            return fix_grid(u,latlon)
        if not Tgrid:
            latlon = self.local_hres_coords[:2]
        else:
            latlon = self.local_hres_coords[2:]
        fu = fix_grid(u,latlon)
        return fu

    def get_hres(self,i,fillna = False):
        ds = self.ds.isel(time = i)
        if fillna:
            ds = ds.fillna(0)
        u,v,T = ds.u,ds.v,ds.T
        u = u.rename(ulat = "lat",ulon = "lon")
        v = v.rename(ulat = "lat",ulon = "lon")
        T = T.rename(tlat = "lat",tlon = "lon")
        return u,v,T

    def get_grid_fixed_lres(self,i,fields):
        ds = self.ds.isel(time =i)
        U = concat(**{field : self.fix_grid(ds[field]) for field in fields})
        return  U

    def get_grid_fixed_hres(self,i,**kwargs):
        u,v, T = self.get_hres(i,**kwargs)
        return self.fix_grid(u,Tgrid= False), self.fix_grid(v,Tgrid= False), self.fix_grid(T,Tgrid= True)
    def build_forcing_mask(self,):
        mask = no_nan_input_mask(self.wet_mask,self.half_spread,lambda x: x==0)
        mask = xr.where(mask==0,1,0)
        return mask

    def build_wet_mask(self,u,T):
        def coarsen_hres_mask(u):
            mask = no_nan_input_mask(u,self.coarse_graining_spread)
            # vals = mask.values[::self.sigma,::self.sigma]
            # mask = xr.DataArray(
            #     data = vals,
            #     dims = ["lat","lon"],
            #     coords = dict(
            #         lat =mask.lat.values[::self.sigma],
            #         lon =mask.lon.values[::self.sigma],
            #     )
            # )
            return mask.coarsen(lat = self.sigma,lon = self.sigma,boundary = "trim").mean()
        umask = coarsen_hres_mask(u)
        tmask = coarsen_hres_mask(T).interp(lat = umask.lat.values,lon = umask.lon.values)
        mask = (umask + tmask)/2
        
        return mask

    def init_coarse_graining(self,):
        u,_,T=self.get_hres(0)
        u,T = self.fix_grid(u,Tgrid = False),self.fix_grid(T,Tgrid = True)
        cgu = coarse_graining_2d_generator(u,self.sigma,wetmask = self.wetmask_flag)
        cgt = coarse_graining_2d_generator(T,self.sigma,wetmask = self.wetmask_flag)
        self.coarse_grain =  (cgu,cgt)


        self.wet_mask = self.build_wet_mask(u,T)

       
        
        
        self.forcing_mask = self.wet_mask # self.build_forcing_mask()


        self.all_land = np.mean(self.forcing_mask.values) > 1 - 1e-2

    def init_coarse_masks(self,):
        u = self.ds.isel(time = 0).u
        u = self.fix_grid(u)
        self.wet_mask = xr.where(no_nan_input_mask(u,0) == 0 ,1,0)
        mask = no_nan_input_mask(u,self.half_spread + 1)
        self.forcing_mask = self.wet_mask# xr.where(mask==0,1,0)
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
    def hres2lres(self,i):
        if not self.initiated:
            self.init_coarse_graining()
            self.initiated = True

        if self.all_land:
            U,F = self.nan_map
            return dict(fields = U, forcings = F)
        u,v,T = self.get_grid_fixed_hres(i,fillna = True)

        F,U = subgrid_forcing(u,v,T,*self.coarse_grain)

        


        M0 = self.wet_mask.sel(**self.final_boundaries) #== 1
        M1 = self.forcing_mask.sel(**self.final_boundaries)# == 1

        F =  F.sel(**self.final_boundaries) #xr.where(M1 , F.sel(**self.final_boundaries), np.nan)
        U = U.sel(**self.final_boundaries) #xr.where(M0 , U.sel(**self.final_boundaries), np.nan)
        return dict(fields = U,forcings = F, field_wetmask = M0,forcing_wetmask = M1)

    def lres2lres(self,i):
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
    def __getitem__(self,i):
        if self.coarse_grain_needed:
            return self.hres2lres(i)
        else:
            return self.lres2lres(i)






def reng_ut_grid_alignment():
    def wet_mask(u):
        u[u==u] = 1
        u[u!=u] = 0
        return u
    u =wet_mask(u)
    T = wet_mask(T)
    y0,y1,x0,x1 = 500-1,700-1,2000+3,3500+3
    Tp = T[y0:y1,x0:x1]
    dx = Tp.shape[1]
    dy = Tp.shape[0]

    ddx = 11
    ddy = 11
    m = np.zeros((ddy,ddx))
    for i,j in itertools.product(range(y0-ddy//2,y0+ddy//2+1),range(x0-ddx//2,x0+ddx//2+1)):
        up = u[i:i+dy,j:j+dx]
        ii,jj = i-(y0-ddy//2),j-(x0-ddx//2)
        m[ii,jj] = np.sum(up*Tp)
        # break
    py,px = np.unravel_index(np.argmax(m.flatten()),m.shape)
    py,px = py - ddy//2, px - ddx//2
