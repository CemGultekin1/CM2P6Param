import itertools
import numpy as np
import xarray as xr



def trim_grid_nan_boundaries(u:np.ndarray,lon = True,lat = True):
    latslice = [0,-1]
    lonslice = [0,-1]
    if lat:
        latcut = 0
        while np.all(np.isnan(u[latcut,:])):
            latslice[0] = latcut
            latcut+=1
        latcut = u.shape[0] - 1
        while np.all(np.isnan(u[latcut,:])):
            latslice[1] = latcut
            latcut-=1
    if lon:
        loncut = 0
        while np.all(np.isnan(u[:,loncut])):
            lonslice[0] = loncut
            loncut+=1
        loncut = u.shape[1] - 1
        while np.all(np.isnan(u[:,loncut])):
            lonslice[1] = loncut
            loncut-=1
    latslice = slice(*latslice)
    lonslice = slice(*lonslice)
    return latslice,lonslice


def ugrid2tgrid(ulat,ulon):
    dulon = ulon[1:] - ulon[:-1]
    dulon = np.append(dulon,dulon[-1])
    dulat = ulat[1:] - ulat[:-1]
    dulat = np.append(dulat,dulat[-1])

    tlon = ulon - dulon/2
    tlat = ulat - dulat/2
    return tlat,tlon
    
def assign_tgrid(ds):
    ulat,ulon = ds.ulat.values,ds.ulon.values
    tlat,tlon = ugrid2tgrid(ulat,ulon)
    ds = ds.assign(tlat = tlat,tlon = tlon)
    return ds

def expand_longitude(ulon,lonmin,lonmax,expand):
    ulon = np.sort((ulon + 180)%360 - 180)
    while ulon[expand]>lonmin:
        ulon = np.concatenate([ulon-360 , ulon])
    while ulon[-1 - expand]<lonmax:
        ulon = np.concatenate([ulon,ulon + 360])
    return np.sort(ulon)

def bound_grid(lat,lon,latmin,latmax,lonmin,lonmax,expand):
    
    I = np.where(lon>lonmin)[0]
    ilon0 = I[0] - expand

    I = np.where(lon<lonmax)[0]
    ilon1 = I[-1] + expand

    lon0,lon1 = lon[[ilon0,ilon1]]


    I = np.where(lat>latmin)[0]
    if len(I)>0:
        ilat0 = np.maximum(I[0] - expand,0)
    else:
        ilat0 = 0

    I = np.where(lat<latmax)[0]

    if len(I)>0:
        ilat1 = np.minimum(I[-1] + expand,len(lat)-1)
    else:
        ilat1 = len(lat)-1

    lat0 = lat[ilat0]
    lat1 = lat[ilat1]

    lat = lat[ilat0:ilat1+1]
    lon = lon[ilon0:ilon1+1]
    return lat,lon,lat0,lat1,lon0,lon1

def longitude_rolling(u:xr.DataArray,):
    field = "ulon"
    u = u.assign_coords({field: (u[field] %360)-180})
    lon = u[field].values
    ilon0 = np.argmin(np.abs(lon + 180))
    u = u.roll({field: -ilon0},roll_coords = True)
    u = u.roll({"tlon": -ilon0}, roll_coords = True)
    return u

def fix_grid(u,target_grid):
    lat,lon = target_grid
    u = fix_latitude(u,lat)
    return fix_longitude(u,lon)

def fix_latitude(u,target_grid):
    return u.sel(lat = slice(*target_grid[[0,-1]]))

def larger_longitude_grid(ulon):
    m = int(1e5)
    ulon = np.round( ulon * m ).astype(int)
    ulon = (ulon +180*m)%(360*m) - (m*180)
    ulon = np.round(ulon).astype(int)
    ulon = np.unique(ulon)
    ulon = np.sort(ulon)/m
    ulon = np.concatenate([ulon - 360,ulon,ulon+360])
    return ulon

def fix_longitude(u,target_grid,):
    u = normalize_longitude(u)
    grid = u.lon.values
    larger_grid = np.concatenate([grid - 360,grid,grid + 360])
    def locate(*locs):
        ilocs = []
        for loc in locs:
            i = np.argmin(np.abs(loc - larger_grid))
            ilocs.append(i)
        return ilocs
    ilon0,ilon1,ilon0_,ilon1_ = locate(*target_grid[[0,-1]],*grid[[0,-1]])
    dilon0 = np.maximum(ilon0_ - ilon0,0)
    dilon1 = np.maximum(ilon1 - ilon1_,0)
    u = u.pad(lon = (dilon0,dilon1),mode="wrap").compute()
    
    ilon0__ = ilon0_-dilon0
    ilon1__ = ilon1_+dilon1+1
    new_grid = larger_grid[ilon0__:ilon1__]
    u = u.assign_coords(lon = new_grid)
    u = u.isel(lon = slice(ilon0 - ilon0__,ilon1 - ilon0__ + 1))
    return u

def longitude_overlaps(u:xr.DataArray,):
    lon = u["lon"]
    m = int(1e5)
    lon = np.round(m*lon).astype(int)
    nlon = np.round(((lon + 180*m) %(360*m))-180*m).astype(int)
    uniqs,cts = np.unique(nlon,return_counts = True)
    if np.all(cts == 1):
        return u 
    overlaps = uniqs[cts > 1]
    uval = u.values
    cuval = np.empty(uval.shape)
    cuval[:] = uval[:]
    removeindices = []
    for overlap in overlaps:
        I = np.where(nlon == overlap)[0]
        cols = uval[:,I]
        nonan = np.sum((cols == cols).astype(int),axis = 1)
        allnonan = nonan > 0
        cols[cols!=cols] = 0
        newcols = np.ones(cols.shape[0])*np.nan
        newcols[allnonan] = np.sum(cols[allnonan,:],axis = 1)/nonan[allnonan]
        cuval[:,I[0]] = newcols
        removeindices.append(I[1:])
    removeindices = np.concatenate(removeindices)
    mask = np.ones(uval.shape[1],dtype = bool)
    mask[removeindices] = False
    uval = uval[:,mask]
    lonval = u["lon"][mask]
    output_u = xr.DataArray(data = uval,dims=["lat","lon"],coords = dict(lat = u["lat"].values,lon = lonval))
    return output_u


def normalize_longitude(u:xr.DataArray,):
    u = longitude_overlaps(u)
    u = u.assign_coords({"lon": ((u["lon"] + 180) %360)-180})
    lon = u["lon"].values
    ilon0 = np.argmin(np.abs(lon + 180))
    u = u.roll({"lon": -ilon0},roll_coords = True)
    return u 

def make_divisible_by_grid(ds,sigma,*boundary):
    lat = boundary[:2]
    lon = boundary[2:]
    lats = ds.lat.values[::sigma]
    lons = ds.lon.values[::sigma]
    lons = expand_longitude(lons,lon[0],lon[1],0)
    b = [0]*4
    def getnearest(arr,el,off):
        i = np.argmin(np.abs(el - arr))
        ioff = np.minimum(np.maximum(i+off,0),len(arr)-1)
        ioff = int(ioff)
        val =  arr[ioff]
        return val
        
    b[2] = getnearest(lons,lon[0],-1)
    b[3] = getnearest(lons,lon[1],+1)
    b[0] = getnearest(lats,lat[0],-1)
    b[1] = getnearest(lats,lat[1],+1)
    return b

def boundary2kwargs(*boundary):
    return {'lon':slice(boundary[2],boundary[3]),'lat':slice(boundary[0],boundary[1])}
def equispace(vec,n):
    x = np.linspace(0,len(vec)-1,n +1 ) 
    doubles = np.empty((n,2))
    for i in range(n):
        doubles[i,:] = x[i]-1,x[i+1]+1
    doubles = np.round(doubles).astype(int)

    doubles = np.maximum(doubles,0)
    doubles = np.minimum(doubles,len(vec)-1).astype(int)

    return doubles
def divide2equals(lats,lons,nlat,nlon,*boundary):
    lats,lons,_,_,_,_ = bound_grid(lats,lons,*boundary,0)
    xlats = equispace(lats,nlat)
    xlons = equispace(lons,nlon)

    z = {}
    lat1 = lats[-1]
    lon1 = lons[-1]
    for i,j in itertools.product(range(nlat),range(nlon)):
        lat0,lat1 = xlats[i]
        lon0,lon1 = xlons[j]
        bds = [*lats[[lat0,lat1]],*lons[[lon0,lon1]]]
        z[(i,j)] = tuple(bds)
    return z
    

