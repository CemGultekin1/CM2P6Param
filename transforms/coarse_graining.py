import itertools
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
        return  x.coarsen(**self._coarse_specs).mean()

class greedy_coarse_grain(plain_coarse_grain):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._coarse_graining_wet_density = super().__call__(self.grid.wet_mask)
    def __call__(self,x,greedy = True):
        if greedy:
            return super().__call__(x.fillna(0)*self.grid.wet_mask)/self._coarse_graining_wet_density
        else:
            return super().__call__(x)

class filtering(base_transform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._norm = None
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


class gcm_filtering(filtering):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        n_steps = kwargs.get('n_steps')
        self._no_area_gcm = gcm.Filter(**filter_specs(self.sigma,self.grid,area_weighted=False,wet_masked=False,n_steps = n_steps))
        self._gcm = gcm.Filter(**filter_specs(self.sigma,self.grid,area_weighted=False,wet_masked=False,tripolar = False,n_steps = n_steps))
    def base_filter(self,x):
        return self._no_area_gcm.apply(x,dims = self.dims)
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
        'n_steps' : n_steps
    }
    return specs