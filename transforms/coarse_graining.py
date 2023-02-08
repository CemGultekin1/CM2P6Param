import numpy as np
import gcm_filters as gcm
import xarray as xr
from scipy.ndimage import gaussian_filter
class base_transform:
    def __init__(self,sigma,grid,*args,dims = 'lat lon'.split(),**kwargs):
        self.sigma = sigma
        self.grid = grid
        self.dims = dims

class plain_coarse_grain(base_transform):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self._coarse_specs = dict({axis : self.sigma for axis in self.dims},boundary = 'trim')
    def __call__(self,x):
        landmask = xr.where(np.isnan(x),1,0)
        forcing_coarse = x.coarsen(**self._coarse_specs).mean()
        coarse_landmask = landmask.coarsen(**self._coarse_specs).mean()
        forcing_coarse = xr.where(coarse_landmask>0,np.nan,forcing_coarse)
        return  forcing_coarse

class greedy_coarse_grain(plain_coarse_grain):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def __call__(self,x,greedy = True):
        if greedy:
            wet_mask = xr.where(np.isnan(x),0,1)
            forcing_coarse = x.fillna(0).coarsen(**self._coarse_specs).mean()
            wet_mask_coarse = wet_mask.coarsen(**self._coarse_specs).mean()
            return forcing_coarse/wet_mask_coarse
        else:
            return super().__call__(x)

class filtering(base_transform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._norm = None
        self._wet_mask = None
    @property
    def norm(self,):
        if self._norm is not None:
            return self._norm
        else:
            return self.get_norm()
    def get_norm(self,):
        return self.base_filter(self.grid.area)
    def base_filter(self,x):
        return 
    def filter(self,x):
        return 
    def __call__(self,x):
        return self.filter(x)/self.norm 
    @property
    def wet_density(self,):
        if self._wet_mask is None:
            return self.filter(self.grid.area*self.grid.wet_mask)/self.filter(self.grid.area)*self.grid.wet_mask
        else:
            return self._wet_mask

class gcm_filtering(filtering):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        n_steps = kwargs.get('n_steps',None)
        self._gcm = gcm.Filter(**filter_specs(self.sigma,self.grid,area_weighted=False,wet_masked=True,n_steps = n_steps))
    def base_filter(self,x):
        return self._gcm.apply(x,dims = self.dims)
    def filter(self,x):
        return self._gcm.apply(self.grid.area*x,dims =  self.dims)

class scipy_filtering(filtering):
    def filter(self,x):
        return xr.apply_ufunc(\
            lambda data: gaussian_filter(data, self.sigma/2, mode='wrap'),\
            self.grid.area*x,dask='parallelized', output_dtypes=[float, ])
    def base_filter(self,x):
        return xr.apply_ufunc(\
            lambda data: gaussian_filter(data, self.sigma/2, mode='wrap'),\
            x,dask='parallelized', output_dtypes=[float, ])

class greedy_scipy_filtering(filtering):
    def filter(self,x):
        wet_mask = xr.where(np.isnan(x),0,1)
        output = xr.apply_ufunc(\
            lambda data: gaussian_filter(data, self.sigma/2, mode='wrap'),\
            x.fillna(0)*self.grid.area*wet_mask,dask='parallelized', output_dtypes=[float, ])
        output = xr.where(wet_mask,output,np.nan)
        return output
    def base_filter(self,x):
        return xr.apply_ufunc(\
            lambda data: gaussian_filter(data, self.sigma/2, mode='wrap'),\
            x*self.grid.wet_mask,dask='parallelized', output_dtypes=[float, ])



def filter_specs(sigma,grid, area_weighted = False, wet_masked= False,tripolar = False,n_steps = 16):
    wetmask,area = grid.wet_mask.copy(),grid.area.copy()
    filter_scale = sigma/2*np.sqrt(12)
    dx_min = 1
    grid_vars = dict(area = area,wet_mask = wetmask)
    if tripolar:
        grid_type = gcm.GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED
    else:
        grid_type = gcm.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED
    if area_weighted and not wet_masked:
        grid_type = gcm.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED
        grid_vars['wet_mask'] = wetmask*0 + 1
    elif not area_weighted and wet_masked:
        grid_type = gcm.GridType.REGULAR_WITH_LAND
        grid_vars.pop('area')
    elif not area_weighted and not wet_masked:
        grid_type = gcm.GridType.REGULAR
        grid_vars = dict()
    specs = {
        'filter_scale': filter_scale,
        'dx_min': dx_min,
        'filter_shape':gcm.FilterShape.GAUSSIAN,
        'grid_type':grid_type,
        'grid_vars':grid_vars,
    }
    if n_steps is not None:
        specs['n_steps'] = n_steps
    return specs