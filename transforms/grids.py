import itertools
import numpy as np
import xarray as xr

EARTH_RADIUS = 6.371*1e6 # in meters

def lose_tgrid(ds):
    vrns = list(ds.data_vars)
    data_vars = {vrn: (
        ['lat','lon'],ds[vrn].values
    ) for vrn in vrns}
    coords = {'lat':ds['ulat'].values,'lon':ds['ulon'].values}
    return xr.Dataset(data_vars=data_vars,coords = coords)


def trim_expanded_longitude(sfd,expansion = 5):
    # lonslice = longitudinal_nan_cut_values(sfd)
    lonslice = slice(expansion,-expansion)
    # print('lonslice:\t',lonslice)
    sfd = sfd.isel(ulon = lonslice,tlon = lonslice,)
    uflag,uslice = assert_periodic_sufficiency(sfd.ulon.values)
    tflag,tslice = assert_periodic_sufficiency(sfd.tlon.values)
    assert uflag
    assert tflag
    sfd = sfd.isel(ulon = uslice,tlon = tslice)
    ulon = sfd.ulon.values
    tlon = sfd.tlon.values
    def normalize(lon):
        return (lon + 180)%360 - 180
    pulon = normalize(ulon)
    ptlon = normalize(tlon)
    ulon0 = np.argmin(np.abs(pulon  + 180))
    tlon0 = np.argmin(np.abs(ptlon + 180))
    ilon0 = np.minimum(ulon0,tlon0).astype(int)
    sfd = sfd.roll({"ulon": -ilon0},roll_coords = True)
    sfd = sfd.roll({"tlon": -ilon0},roll_coords = True)
    sfd["ulon"] = normalize(sfd["ulon"].values)
    sfd["tlon"] = normalize(sfd["tlon"].values)
    return sfd

def assert_periodic_sufficiency(lon,):
    M = int(1e5)
    lon0 = int((lon[0] + 360)*M)
    lon1 = int(lon[-1]*M)
    sufficiency = lon0 <= lon1
    if not sufficiency:
        return False, None
    plon = (lon*M).astype(int)
    p0 = plon[0] + int(360*M)
    I = np.where(p0 <= plon )[0]
    assert p0 > plon[I[0]-1]
    return True,slice(0,I[0]-1)
    
def longitudinal_nan_cut_values(sfd):
    def lon_cut(var):
        val = var.values
        i = 0
        while np.all(np.isnan(val[:,i])):
            i+=1
        i0 =i 

        i = val.shape[1]-1
        while np.all(np.isnan(val[:,i])):
            i-=1
        i1 = i - val.shape[1]
        return i0,i1
    lon0,lon1 = -1,1
    for key in sfd.data_vars:
        i0,i1 = lon_cut(sfd[key])
        lon0 = np.maximum(i0,lon0)
        lon1 = np.minimum(i1,lon1)
    lonslice = slice(lon0,lon1)
    return lonslice



def forward_difference(x:xr.DataArray,dx:xr.DataArray,field):
    dx = x.diff(field)/dx
    f0 = x[field][0]
    dx = dx.pad({field : (1,0)},constant_values = np.nan)
    dxf = dx[field].values
    dxf[0] = f0
    dx[field] = dxf
    return dx

def ugrid2tgrid(u:xr.DataArray,v:xr.DataArray,ugrid:xr.Dataset,tgrid:xr.Dataset):
    uval = u.values
    vval = v.values
    dlat = ugrid.dy.values
    dlon = ugrid.dx.values

    udlat = uval*dlat

    vdlon = vval*dlon
    udlat = (udlat[:,1:] + udlat[:,:-1])/(2*dlat[:,1:])
    vdlon = (vdlon[:,1:] + vdlon[:,:-1])/(dlon[:,1:]+dlon[:,:-1])
    udlat = np.concatenate([np.zeros((udlat.shape[0],1)),udlat],axis =1)
    vdlon = np.concatenate([np.zeros((vdlon.shape[0],1)),vdlon],axis =1)
    coords = dict(dims = ['lat','lon'],
        coords = dict(
            lat = tgrid.lat.values,
            lon = tgrid.lon.values,
        )
    )
    ut = xr.DataArray(
        data = udlat,
        **coords
    )

    vt = xr.DataArray(
        data = vdlon,
        **coords
    )
    return ut,vt



def get_separated_grid_vars(grid:xr.Dataset,prefix = "u"):
    lon = grid[f"{prefix}lon"].values
    radlon = lon/180*np.pi
    lat = grid[f"{prefix}lat"].values
    radlat = lat/180*np.pi
    dlat = radlat[1:] - radlat[:-1]
    dlon = radlon[1:] - radlon[:-1]
    dlat = np.concatenate([dlat,dlat[-1:]])
    dlon = np.concatenate([dlon,dlon[-1:]])
    area_y =  dlat
    area_x = np.cos(radlat)*dlon 
    

    area_x = xr.DataArray(
        data = area_x,
        dims = [f"{prefix}lon"],
        coords = {
            f"{prefix}lon" : lon
        },
        name = 'area_lon',
    )
    area_y = xr.DataArray(
        data = area_y,
        dims = [f"{prefix}lat"],
        coords = {
            f"{prefix}lat" : lat
        },
        name = 'area_lat',
    )
   
    return xr.merge([area_x,area_y])



def get_grid_vars(grid:xr.Dataset,):
    vars = []
    for prefix,wetvar in zip('u t'.split(),'u T'.split()):
        dlon = f"dx{prefix}"
        dlat = f"dy{prefix}"
        name_lon = f"{prefix}lon"
        name_lat = f"{prefix}lat"
        kwargs = dict(
            coords = dict(
                lat = grid[name_lat].values,lon = grid[name_lon].values
            ),
        )
        dx = xr.DataArray(
            data = grid[dlon].values,
            **kwargs,name = 'dx'
        )
        dy = xr.DataArray(
            data = grid[dlat].values,
            **kwargs,name = 'dy'
        )
        area = dx*dy
        area.name = 'area'
        wetmask = xr.DataArray(
            data = xr.where(np.isnan(grid[wetvar]), 0,1).values,
            **kwargs,name = 'wetmask'
        )
        var1 =  xr.merge([dx,dy,area,wetmask])
        vars.append(var1)
    return vars


def logitudinal_expansion(u:xr.DataArray,expansion,prefix = ''):
    slon = f"{prefix}lon"
    slat = f"{prefix}lat"

    lon =u[slon].values
    uval = u.values
    
    lon0 = lon[:expansion] + 360
    lon1 = lon[-expansion:]- 360

    u0 = uval[:,:expansion]
    u1 = uval[:,-expansion:]


    newlon = np.concatenate([lon1,lon,lon0])
    newuval = np.concatenate([u1,uval,u0],axis = 1)
    return xr.DataArray(
        data = newuval,
        dims = [slat,slon],
        coords = {
            slat : u[slat].values,
            slon : newlon,
        }
    )

def logitudinal_expansion_dataset(ds:xr.Dataset,expansion,prefix = ''):
    names = list(ds.data_vars)
    data_vars = []
    for name in names:
        u = ds[name]
        slon = f"{prefix}lon"
        slat = f"{prefix}lat"
        lon =u[slon].values
        uval = u.values
        
        lon0 = lon[:expansion] + 360
        lon1 = lon[-expansion:]- 360

        u0 = uval[:,:expansion]
        u1 = uval[:,-expansion:]


        newlon = np.concatenate([lon1,lon,lon0])
        newuval = np.concatenate([u1,uval,u0],axis = 1)
        data_vars.append(xr.DataArray(
            data = newuval,
            dims = [slat,slon],
            coords = {
                slat : u[slat].values,
                slon : newlon,
            },
            name = name
        ))
    return xr.merge(data_vars)



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


# def ugrid2tgrid(ulat,ulon):
#     dulon = ulon[1:] - ulon[:-1]
#     dulon = np.append(dulon,dulon[-1])
#     dulat = ulat[1:] - ulat[:-1]
#     dulat = np.append(dulat,dulat[-1])

#     tlon = ulon - dulon/2
#     tlat = ulat - dulat/2
#     return tlat,tlon

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

    I = np.where(lon>=lonmin)[0]
    ilon0 = I[0] - expand

    I = np.where(lon<=lonmax)[0]
    ilon1 = I[-1] + expand

    def get_slice_extremes(lon,ilon0,ilon1):
        if ilon0 > 0 :
            lon0 = lon[ilon0] - (lon[ilon0] -lon[ilon0-1] )/2
        else:
            lon0 = lon[ilon0] - (lon[ilon0+1] -lon[ilon0] )/2

        if ilon1 < len(lon)-1:
            lon1 = lon[ilon1] + (lon[ilon1+1] -lon[ilon1] )/2
        else:
            lon1 = lon[ilon1] + (lon[ilon1] -lon[ilon1-1] )/2
        return lon0,lon1
    lon0,lon1 = get_slice_extremes(lon,ilon0,ilon1)

    I = np.where(lat>=latmin)[0]
    if len(I)>0:
        ilat0 = np.maximum(I[0] - expand,0)
    else:
        ilat0 = 0

    I = np.where(lat<=latmax)[0]

    if len(I)>0:
        ilat1 = np.minimum(I[-1] + expand,len(lat)-1)
    else:
        ilat1 = len(lat)-1

    lat0,lat1 = get_slice_extremes(lat,ilat0,ilat1)
    
    


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
    dims = list(u.dims)
    dims = [d for d in dims if "lat" in d]
    assert len(dims) == 1
    lat_name = dims[0]
    return u.sel(**{lat_name: slice(*target_grid[[0,-1]])})

def larger_longitude_grid(ulon):
    m = int(1e5)
    ulon = np.round( ulon * m ).astype(int)
    ulon = (ulon +180*m)%(360*m) - (m*180)
    ulon = np.round(ulon).astype(int)
    ulon = np.unique(ulon)
    ulon = np.sort(ulon)/m
    ulon = np.concatenate([ulon - 360,ulon,ulon+360])
    return ulon

def fix_longitude(u,target_grid):
    dims = list(u.dims)
    dims = [d for d in dims if "lon" in d]
    assert len(dims) == 1
    lon_name = dims[0]
    u = normalize_longitude(u,lon_name)
    grid = u[lon_name].values
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
    u = u.pad(**{lon_name :(dilon0,dilon1),"mode":"wrap"}).compute()

    ilon0__ = ilon0_-dilon0
    ilon1__ = ilon1_+dilon1+1
    new_grid = larger_grid[ilon0__:ilon1__]
    u = u.assign_coords(**{lon_name: new_grid})
    u = u.isel(**{lon_name:slice(ilon0 - ilon0__,ilon1 - ilon0__ + 1)})
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


def normalize_longitude(u:xr.DataArray,lon_name):
    u = longitude_overlaps(u)
    u = u.assign_coords({lon_name: ((u[lon_name] + 180) %360)-180})
    lon = u[lon_name].values
    ilon0 = np.argmin(np.abs(lon + 180))
    u = u.roll({lon_name: -ilon0},roll_coords = True)
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
