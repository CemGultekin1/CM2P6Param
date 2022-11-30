import itertools
import gcm_filters as gcm
from gcm_inversion.read_data import get_grid
import numpy as np
from utils.xarray import plot_ds
import xarray as xr

def filter_specs(sigma,grid, area_weighted = False, wetmasked= False):
    wetmask,area = grid.wetmask,grid.area
    filter_scale = sigma/2*np.sqrt(12)
    dx_min = 1
    if area_weighted and wetmasked:
        specs = {
            'filter_scale': filter_scale,
            'dx_min': dx_min,
            'filter_shape':gcm.FilterShape.GAUSSIAN,
            'grid_type':gcm.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
            'grid_vars':{'wet_mask': wetmask,'area': area},
        }
    elif area_weighted and not wetmasked:
        specs = {
            'filter_scale': filter_scale,
            'dx_min': dx_min,
            'filter_shape':gcm.FilterShape.GAUSSIAN,
            'grid_type':gcm.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
            'grid_vars':{'area': area,'wet_mask': wetmask*0 + 1},
        }
    elif not area_weighted and wetmasked:
        specs = {
            'filter_scale': filter_scale,
            'dx_min': dx_min,
            'filter_shape':gcm.FilterShape.GAUSSIAN,
            'grid_type':gcm.GridType.REGULAR_WITH_LAND,
            'grid_vars':{'wet_mask': wetmask},
        }
    else:
        specs = {
            'filter_scale': filter_scale,
            'dx_min': dx_min,
            'filter_shape':gcm.FilterShape.GAUSSIAN,
            'grid_type':gcm.GridType.REGULAR,
        }
    return specs



def spatially_variant_filtering(grid,sigma,i,z):
    '''
    |0---|1---|2   |3---|4---|
    '''
    lspan = 2*sigma
    rspan = 3*sigma + 1
    mid = 2*sigma
    cmid = 2
    latslice,lonslice = [slice(i[j] - lspan,i[j]+rspan) for j in range(2)]
    grid = grid.isel(lat = latslice,lon = lonslice)
    specs = filter_specs(sigma,grid,area_weighted=True,wetmasked=True)
    filter = gcm.Filter(**specs,)
    
    data = grid.area.values*0
    data[mid + z[0],mid + z[1]] = 1
    one_hot = xr.DataArray(
        data = data,
        dims = ['lat','lon'],
        coords = grid.coords
    )
    ohbar = filter.apply(one_hot,dims=['lat','lon'])
    ohbar = ohbar*grid.area/grid.areabar
    cxw = grid.wetmask*ohbar.fillna(0)
    cwargs = dict(lat=sigma,lon=sigma,boundary="trim")
    cxw = cxw.coarsen(**cwargs).mean()/grid.wetmask.coarsen(**cwargs).mean()
    return cxw.values[cmid,cmid]

def straight_filter_weights(sigma,area,wetmask):
    inds = np.arange(-2*sigma,2*sigma+1)
    w = np.exp( - inds**2/(2*(sigma/2)**2))
    w_ = w.reshape([-1,1])@w.reshape([1,-1])
    w = np.zeros((5*sigma+1,5*sigma+1))
    w[:w_.shape[0],:w_.shape[1]] = w_
    # ww = w*0
    # for i in itertools.product(range(sigma),range(sigma)):
    #     ww += np.roll(w,i,axis=(0,1))
    ww = w
    ww = ww*area
    ww = ww * wetmask
    ww = ww/np.sum(ww)
    ww = ww*wetmask[2*sigma,2*sigma]
        
    return ww

def spatially_variant_filtering_prediction(grid,sigma,i):
    lspan = 2*sigma
    rspan = 3*sigma + 1
    latslice,lonslice = [slice(i[j] - lspan,i[j]+rspan) for j in range(2)]
    grid = grid.isel(lat = latslice,lon = lonslice)
    return straight_filter_weights(sigma,grid.area,grid.wetmask)
    
    
class GCM_weights:
    def __init__(self,sigma) -> None:
        self.longexp = 4*sigma
        self.clongexp= self.longexp//sigma
        grid = get_grid(0,longitude_expand = self.longexp)
        self.sigma = sigma
        specs = filter_specs(sigma,grid,area_weighted=False,wetmasked=True)
        areabar = gcm.Filter(**specs).apply(grid.area,dims = ['lat','lon'])
        grid['areabar'] = (areabar.dims, areabar.values)


        self.grid = grid
        specs = filter_specs(sigma,grid,area_weighted=False,wetmasked=False)
        wetbar = gcm.Filter(**specs).apply(1. - grid['wetmask']*1.,dims = ['lat','lon'])
        self.coarse_wet_density = 1 - self.coarsen(wetbar,area_averaged=False)


        slc = slice(self.clongexp, -self.clongexp)
        lats,lons = np.where(self.coarse_wet_density.values[slc,slc] > 1-1e-5)
        ulats,inds = np.unique(lats,return_index= True)
        ulons = lons[inds]
        self.wet_indices = np.stack([ulats,ulons],axis= 1) + self.clongexp
        llats,llons = np.where((self.coarse_wet_density.values[slc,slc] < 1-1e-5 ) *(self.coarse_wet_density.values[slc,slc] > 1e-5))
        self.mixed_indices = np.stack([llats,llons],axis = 1) + self.clongexp
        
    @property
    def nwet(self,):
        return self.wet_indices.shape[0]

    @property
    def nmixed(self,):
        return  self.mixed_indices.shape[0]

    def coarsen(self,x_,area_averaged = False):
        x = x_.fillna(0)
        cx = x.coarsen(**dict(lat = self.sigma,lon = self.sigma),boundary = 'trim').mean()
        if area_averaged:
            wet = xr.where(x == 0,0,1)
            cx = cx/wet.coarsen(**dict(lat = self.sigma,lon = self.sigma),boundary = 'trim').mean()
        return cx

    def lres_loc(self,i):
       return  (i - self.longexp)//self.sigma
    def hres_loc(self,i):
       return  i * self.sigma  + self.longexp
    def loc_wetmask(self,i):
        i = np.array(i)*self.sigma
        zmin = -2*self.sigma 
        zmax = 3*self.sigma + 1
        return self.grid.wetmask.isel(lat = slice(i[0] + zmin, i[0] + zmax),lon = slice(i[1] + zmin,i[1] + zmax)).values
    def apply_mask(self,sv_,i):
        wetmask = self.loc_wetmask(i)
        sv = sv_.copy()
        sv[wetmask < 1] = np.nan
        return sv
    def loc_filters(self,i,navalued = False):
        i = np.array(i)*self.sigma
        nz = 5*self.sigma + 1
        zmin = -2*self.sigma 
        zmax = 3*self.sigma + 1
        sv = np.zeros((nz,nz))
        for z in itertools.product(range(zmin,zmax),range(zmin,zmax)):
            sv[z[0] - zmin, z[1] - zmin] = spatially_variant_filtering(self.grid,self.sigma,i,z)
        if navalued:
            sv = self.apply_mask(sv,i)
        
        return sv
        


        


def main():
    
    sigma = 8
    grid = get_grid(0,sigma)
    
    specs = filter_specs(sigma,grid,area_weighted=False)
    areabar = gcm.Filter(**specs).apply(grid.area,dims = ['lat','lon'])
    grid['areabar'] = (areabar.dims, areabar.values)
    # for i in range(10)
    zmin = -2*sigma 
    zmax = 3*sigma
    nx_ = len(grid.lon)
    ny_ = len(grid.lat)
    nx,ny = nx_ - 4*sigma,ny_ - 4*sigma
    snx,sny = nx//sigma,ny//sigma

    latmid,lonmid = 20,-75
    icenter = [np.argmin(np.abs(grid[key].values - mid)) for key,mid in zip('lat lon'.split(),[latmid,lonmid])]
    m = 4
    hm = m//2
    di = 5*sigma
    latvals,lonvals = [icenter[j] + di*np.arange(-hm,m - hm) for j in range(2)]
    def get_filters(direct = False):
        placed_filters = np.zeros((ny_,nx_))
        # locfilters = dict()
        for num,i in enumerate(itertools.product(latvals,lonvals)):
            print(num,)
            nz = zmax - zmin + 1
            if not direct:
                sv = np.zeros((nz,nz))
                for z in itertools.product(range(zmin,zmax+1),range(zmin,zmax+1)):
                    sv[z[0] - zmin, z[1] - zmin] = spatially_variant_filtering(grid,sigma,i,z)
            else:
                sv = spatially_variant_filtering_prediction(grid,sigma,i)
            latslice,lonslice = [ slice(i_ + zmin , i_ + zmin + nz ) for i_ in i ]
            # np.all(np.isnan(sv))
            placed_filters[latslice,lonslice] = sv
            # locfilters[str(i)] = xr.DataArray(
            #     data = sv,#.reshape([1,1,sv.shape[0],sv.shape[1]]),
            #     dims = ['zlat','zlon'],
            #     coords = dict(
            #         zlat = np.arange(nz),zlon = np.arange(nz)
            #     )
            # )
            # break
        placed_filters = xr.DataArray(
            data = placed_filters,
            dims = ['lat','lon'],
            coords = grid.coords
        )
        placed_filters = xr.where(grid.wetmask,placed_filters,np.nan)
        placed_filters = placed_filters.isel(
            {key: slice(ic - m*di ,ic + m*di ) for key,ic in zip('lat lon'.split(),icenter)}
        )
        return placed_filters
    import matplotlib.pyplot as plt
    get_filters(direct = True).plot()
    plt.savefig('direct_locfilters.png')
    plt.close()
    # get_filters(dircet = True).plot()
    # plt.savefig('locfilters.png')
    # plt.close()
    
    # plot_ds(locfilters,'locfilters',ncols = 3,dims =['zlat','zlon'])
    return 

if __name__=='__main__':
    main()
