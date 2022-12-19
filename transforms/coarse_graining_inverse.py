import itertools
from transforms.coarse_graining import base_transform,filtering, gcm_filtering, greedy_coarse_grain, greedy_scipy_filtering
import numpy as np
import xarray as xr


class inverse_filtering:
    filtering_class = None
    def __init__(self,*args,**kwargs):
        self.coarse_grain = greedy_coarse_grain(*args,**kwargs)
        self.mat_gcm = matmult_gcm_filtering(*args,**kwargs)
        self.filtering  :filtering = self.filtering_class(*args,**kwargs)
        self.coarse_grained_wet_density = self.coarse_grain(self.filtering.wet_density,greedy = False)

    def __call__(self,x):
        return self.mat_gcm(x,inverse = True,wet_density = self.coarse_grained_wet_density)

class inverse_greedy_scipy_filtering(inverse_filtering):
    filtering_class = greedy_scipy_filtering


class inverse_gcm_filtering(inverse_filtering):
    filtering_class = gcm_filtering

def right_inverse_matrix(mat):
    q,r = np.linalg.qr(mat.T)
    return q @ np.linalg.inv(r).T
    
class matmult_gcm_1d(base_transform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        area = self.grid.area.values
        n = len(area)
        self.weights =filter_weights_1d(self.sigma)
        filter_mat = np.zeros((n, n))
        m = len(self.weights)//2

        padded_weights = np.zeros(n)
        padded_weights[:2*m + 1] = self.weights
        padded_weights = np.roll(padded_weights,-m)

        for i in range(n):
            filter_mat[i,:] = np.roll(padded_weights, i)*area
        cfm = filter_mat*1
        for j in range(1,self.sigma):
            cfm[:-j] = cfm[:-j] + filter_mat[j:]
        cfm = cfm[::self.sigma]
        cfm = cfm[:n//self.sigma]
        cfm = cfm/np.sum(cfm,axis = 1,keepdims = True)

        self._matrix = cfm
        self._right_inv_matrix = right_inverse_matrix(self._matrix)
    def __call__(self,x,ax = 0,inverse = False):
        if inverse:
            mat = self._right_inv_matrix
        else:
            mat = self._matrix
        if ax == 0:
            return mat @ x
        else:
            return x @ mat.T
        
class matmult_gcm_filtering(base_transform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        lon_area = self.grid.mean(dim = self.dims[0])
        lat_area = self.grid.mean(dim = self.dims[1])
        self._lonfilt = matmult_gcm_1d(self.sigma,lon_area,**kwargs)
        self._latfilt = matmult_gcm_1d(self.sigma,lat_area,**kwargs)
        self._full_wet_density = self.base__call__(self.grid.wet_mask)
        

    def base__call__(self,x,inverse = False):
        
        xv = x.fillna(0).values
        xvv = self._latfilt(self._lonfilt(xv,ax=1,inverse = inverse),ax = 0,inverse = inverse)
        dims = self.dims
        if inverse:
            return xr.DataArray(
                data = xvv,
                dims = dims,
                coords = {
                    key : self.grid[key].values for key in dims
                }
            )
        else:
            return xr.DataArray(
                data = xvv,
                dims = dims,
                coords = {
                    key : self.grid[key].coarsen(**{key : self.sigma,'boundary' : 'trim'}).mean().values for key in dims
                }
            )
    def __call__(self,x,inverse = False,wet_density = None):
        if wet_density is None:
            wet_density = self._full_wet_density
        if inverse:
            x = x*wet_density

        cx = self.base__call__(x,inverse = inverse)
        if not inverse:
            cx = cx/wet_density
        if inverse:
            cx = xr.where(self.grid.wet_mask,cx,np.nan)
        return cx


def filter_weights_1d(sigma):
    fw = filter_weights(sigma)
    fw = fw[:,(5*sigma + 1)//2]
    fw = fw/sum(fw)
    return fw

def filter_weights(sigma):
    inds = np.arange(-2*sigma,2*sigma+1)
    w = np.exp( - inds**2/(2*(sigma/2)**2))
    w_ = w.reshape([-1,1])@w.reshape([1,-1])
    w = np.zeros((5*sigma+1,5*sigma+1))
    w[:w_.shape[0],:w_.shape[1]] = w_
    ww = w*0
    for i in itertools.product(range(sigma),range(sigma)):
        wm = np.roll(w,i,axis=(0,1))
        ww += wm
    sww = np.sum(ww)
    ww = ww/sww
    return ww

