
import itertools
import gcm_filters as gcm
from gcm_inversion.quadratic_program import optimize
from gcm_inversion.read_data import get_grid
import numpy as np
from utils.xarray import plot_ds
import xarray as xr
from scipy.ndimage import gaussian_filter

def filter_weights(sigma,area,wetmask):
    inds = np.arange(-2*sigma,2*sigma+1)
    w = np.exp( - inds**2/(2*(sigma/2)**2))
    w_ = w.reshape([-1,1])@w.reshape([1,-1])
    w = np.zeros((5*sigma+1,5*sigma+1))
    w[:w_.shape[0],:w_.shape[1]] = w_
    ww = w*0
    for i in itertools.product(range(sigma),range(sigma)):
        wm = np.roll(w,i,axis=(0,1))*area*wetmask*wetmask[2*sigma + i[0],2*sigma + i[1]]
        swm = np.sum(wm)
        # if swm==0:
        #     continue
        # wm = wm/swm
        # wm = wm*wetmask
        # swm = np.sum(wm)
        if swm==0:
            continue
        wm = wm/swm
        ww += wm
    sww = np.sum(ww)
    if sww == 0 :
        return np.zeros((5*sigma+1,5*sigma+1))
    else:
        ww = ww/sww
    # ww = ww*wetmask[2*sigma,2*sigma]
    return ww

def grid_patch(grid,sigma,i,di = [0,0]):
    lspan = 2*sigma
    rspan = 3*sigma + 1
    # np.arange(-di - 2,di+1 + 3)*sigma
    latslice,lonslice = [slice(i[j] - di[0] - lspan,i[j] + di[1] +rspan) for j in range(2)]
    grid = grid.isel(lat = latslice,lon = lonslice)
    return grid
def spatially_variant_filter(grid,sigma,i):
    grid = grid_patch(grid,sigma,i)
    return filter_weights(sigma,grid.area.values,grid.wetmask.values)
def basic_filter_weights(sigma):
    lspan = 2*sigma
    rspan = 3*sigma + 1
    totspan = lspan + rspan
    area = np.ones((totspan,totspan))
    return filter_weights(sigma,area,area)
def generate_1d_basis(sigma):
    '''
    5*sigma + 1 many values
    s,s/2,s/2,s/4,s/4,1,s/4,s/4,s/2,s/2,s
    '''
    w =basic_filter_weights(sigma)
    w = w[2*sigma,:]
    s = sigma
    # parts = np.array([s,s/2,s/4,s/4,s/4,s/4,1,s/4,s/4,s/4,s/4,s/2,s]).astype(int)
    parts = np.array([s,s,s/2,1,s/2,s,s]).astype(int)
    # parts = np.ones(5*sigma +1).astype(int)
    # # parts = np.array([s,s, s/2,1,s/2, s,s]).astype(int)
    parts = np.concatenate([[0],np.cumsum(parts)])
    basis = []
    for i0,i1 in zip(parts[:-1],parts[1:]):
        cw = w.copy()
        cw[:i0] = 0
        cw[i1:] = 0
        basis.append(cw)
    basis = np.stack(basis,axis= 0) #np.eye(5*sigma + 1) #
    basis[-1] = w
    q,_ = np.linalg.qr(basis.T)
    # q = 
    return q.T
    # return q[:11,:]
def project_filter(ftr,basis):
    return basis@ftr@basis.T

def restricted_grid_values(ds,icenter,grid,sigma):
    latvals,lonvals = [ic + np.arange(-di,di+1)*sigma for di,ic in zip(ds,icenter)]

    latvals1,lonvals1 = [ic + np.arange((-di - 2)*sigma ,(di  +1 + 3)*sigma + 1) for di,ic in zip(ds,icenter)]

    clat,clon = [grid[key].coarsen({key:sigma},boundary = 'trim').mean().values for key in 'lat lon'.split()]
    clat,clon = [c[ic//sigma - di : ic//sigma + di +1] for c,di,ic in zip([clat,clon],ds,icenter)]

    lat,lon = [grid[key].values[ic - (di + 2)*sigma : ic + (di+1 + 3)*sigma +1] for key,di,ic in zip('lat lon'.split(),ds,icenter)]
    return (latvals1,lonvals1),(latvals,lonvals),(clat,clon),(lat,lon)


def single_basis_vector_forward_filter(cn,n,bvec,sigma):
    z = np.zeros((cn,n))
    for i in range(cn):
        csl = slice(i*sigma,i*sigma + 5*sigma + 1)
        z[i,csl] = bvec
    eyey = z@z.T
    ortherr = np.diag(np.diag(eyey)) - eyey
    if not np.all(np.abs(ortherr)<1e-9):
        print(cn,n,sigma)
        import matplotlib.pyplot as plt
        for i in range(len(z)):
            plt.plot(z + i)
        plt.savefig('non_orth_ex.png')
        raise Exception
    return z
def forward_filter_mat(cn,n,basis,sigma):
    nb = basis.shape[0]
    z = []
    for i in range(nb):
        z.append(single_basis_vector_forward_filter(cn,n,basis[i],sigma))
    z = np.stack(z,axis = 0)
    return z

# def right_inverse_filter_mat(filter_mat,):
#     shp = filter_mat.shape
#     filter_mat = filter_mat.reshape([shp[0]*shp[1],-1])
#     q,r = np.linalg.qr(filter_mat.T)
#     q,r = np.linalg.qr(filter_mat,mode = 'reduced')
#     return (q@np.linalg.inv(r).T).reshape(shp)

def left_inverse_single_filter_mat(filter_mat,rcond_lim = 1e-2):
    u,s,vh = np.linalg.svd(filter_mat,full_matrices = False)
    rs =s/s[0]
    I = rs > rcond_lim
    u = u[:,I]
    s = s[I]
    vh = vh[I,:]
    invfilter_mat = u @ np.diag(1/s) @ vh
    projection_mat = u
    return invfilter_mat,projection_mat
def left_inverse_filter_mat(flat,rcond_lim = 1e-4):
    # shp = flat.shape
    # invflat,proj = left_inverse_single_filter_mat(flat.reshape([-1,shp[2]]),rcond_lim= rcond_lim)
    # proj = proj.reshape([shp[0],shp[1],-1])
    # invflat = invflat.reshape(shp)#.transpose((0,2,1))
    # return invflat,proj
    invflat =[]
    projflat = []
    for flati in flat:
        inv_,proj_ = left_inverse_single_filter_mat(flati,rcond_lim=rcond_lim)
        invflat.append(inv_)
        projflat.append(proj_)
    return np.stack(invflat,axis = 0),np.stack(projflat,axis =0)

def left_inverse(locf,flat,flon,rcond_lim = 1e-9):
    invlat,projlat = left_inverse_filter_mat(flat,rcond_lim = rcond_lim)
    invlon,projlon = left_inverse_filter_mat(flon,rcond_lim = rcond_lim)
    for z,(plat,plon) in enumerate(itertools.product(projlat,projlon)):
        locfz = locf[z]
        locfz = plat@(plat.T@locfz)
        locfz = (locfz@plon)@plon.T
        locf[z] = locfz
    invlocf = locf.copy()
    norms = np.sum(invlocf**2,axis=0,keepdims = True)
    mask = norms < rcond_lim
    invlocf = invlocf*(1 - mask)
    norms[mask] = 1
    invlocf = invlocf/norms
    return locf,invlocf,invlat,invlon

def coarse_grain_by_weights(x,locfilters,filter_lat,filter_lon):
    cx = None
    for z,(i,j) in enumerate(itertools.product(range(filter_lat.shape[0]),range(filter_lon.shape[0]))):
        flati = filter_lat[i]
        flonj = filter_lon[j]
        xij = (flati@x)@flonj.T
        xij = locfilters[z]*xij
        if cx is None:
            cx = xij
        else:
            cx += xij
    return cx

def invert_filtering(cx,invlocfcoeff,invflat,invflon):
    hx0 = None
    for z,(invflati,invflonj) in enumerate(itertools.product(invflat,invflon)):
        cxx = cx*invlocfcoeff[z]
        hx0_ = (invflati.T@cxx)@invflonj
        if hx0 is None:
            hx0 = hx0_
        else:
            hx0 += hx0_
    return hx0


def get_minimal_spread(flat,flon):
    def middle_minimal_(flat):
        x = np.sum(np.abs(flat)>0,axis = (1,2))
        I = np.where(x == np.amin(x))[0]
        I = np.sort(I)
        return I[len(I)//2]
    return middle_minimal_(flat), middle_minimal_(flon)

def initial_solution(cx,invlocfilters,invflat,invflon,flatmid,flonmid):
    i = invflon.shape[0]*flatmid + flonmid
    cx0 =  (invflat[flatmid].T@(cx*invlocfilters[i]))@invflon[flonmid]
    nlat = invflat.shape[2]
    nlon = invflon.shape[2]
    nbasis = invflon.shape[0]*invflat.shape[0]
    shp = (nbasis,nlat,nlon)
    us = np.zeros(shp)
    us[i] = cx0
    return us

def main():
    sigma = 4
    basis = generate_1d_basis(sigma)
    
    grid = get_grid(0,sigma)
    # grid['wetmask'] = grid.wetmask*0 + 1
    grid['area'] = grid.area*0 + 1
    latmid,lonmid = 20,-75
    icenter = [np.argmin(np.abs(grid[key].values - mid)) for key,mid in zip('lat lon'.split(),[latmid,lonmid])]
    icenter = [(ic//sigma)*sigma for ic in icenter]
    cds = [5,5]
    ds = [d*sigma  for d in cds]
    ds[1] += sigma
    latloninds1,latloninds, clatlon,latlon = restricted_grid_values(cds,icenter,grid,sigma)

    ny1,nx1 = [len(ll) for ll in latlon]
    cny1,cnx1 = [len(ll) for ll in clatlon]
    filter_lat = forward_filter_mat(cny1,ny1,basis,sigma)
    filter_lon = forward_filter_mat(cnx1,nx1,basis,sigma)
    
    flatmid,flonmid = get_minimal_spread(filter_lat,filter_lon)

    
    print([x.shape for x in [filter_lat,filter_lon]])

    locfilters = []
    for num,i in enumerate(itertools.product(*latloninds)):
        if num % 300 == 0 :
            print(num)
        sv = spatially_variant_filter(grid,sigma,i)
        locfilters.append(project_filter(sv,basis).reshape([-1]))
    locfilters = np.stack(locfilters,axis= 1)
    locfilters = locfilters.reshape([-1,cds[0]*2+1,cds[1]*2+1])

    def build_cos_signal(ns,nper1,nper2):
        nx,ny = ns
        ys,xs = np.meshgrid(np.linspace(0,1,nx),np.linspace(0,1,ny))
        ln = np.linspace(0,1,100)
        lnx, lny = ln, (np.cos(ln*4*np.pi) + 2)/4
        z = ys*0
        for i,j in itertools.product(range(ys.shape[0]),range(ys.shape[1])):
            y,x = ys[i,j],xs[i,j]
            z[i,j] = np.amin((lnx - x)**2 + (lny - y)**2)
        z = z.T 
        return z

    x = build_cos_signal([ny1,nx1],1,2)

    

    
    
    cx = coarse_grain_by_weights(x,locfilters,filter_lat,filter_lon)

    locfilters,invlocfilters, invflat,invflon = left_inverse(locfilters,filter_lat,filter_lon,rcond_lim = 1e-5)
    

    print((locfilters.shape,invlocfilters.shape),\
        (invflat.shape,filter_lat.shape),\
        (invflon.shape,filter_lon.shape))

    cx0 = initial_solution(cx,invlocfilters,invflat,invflon,flatmid,flonmid)


    # cx0 =  xr.DataArray(
    #     data = cx0,
    #     dims = ['lat','lon'],
    #     coords = dict(lat = latlon[0],lon= latlon[1])
    # )


    cx1 = optimize(cx0,filter_lat,filter_lon,locfilters, sigma)

    return
    ccx0 = (invflat[flatmid]@cx0)@invflon[flatmid].T
    
    cx0 =  xr.DataArray(
        data = cx0,
        dims = ['lat','lon'],
        coords = dict(lat = latlon[0],lon= latlon[1])
    )

    x =  xr.DataArray(
        data = x,
        dims = ['lat','lon'],
        coords = dict(lat = latlon[0],lon= latlon[1])
    )

    ccx0 = xr.DataArray(
        data = ccx0,
        dims = ['clat','clon'],
        coords = dict(clat = clatlon[0],clon= clatlon[1])
    )

    cx = xr.DataArray(
        data = cx,
        dims = ['clat','clon'],
        coords = dict(clat = clatlon[0],clon= clatlon[1])
    )


    plot_ds({'x':x,'cx0':cx0},'hres',ncols = 2,dims = ['lat','lon'])
    plot_ds({'cx':cx,'ccx0':ccx0},'lres',ncols = 2,dims = ['clat','clon'])
    return 
    invlocfs = dict()
    for i,invlocfi in enumerate(invlocfilters):
        if i not in np.arange(invlocfilters.shape[0]//5)*5:
            continue
        invlocfs[str(i)] = xr.DataArray(
            data = invlocfi,
            dims = ['clat','clon'],
            coords = dict(clat = clatlon[0],clon= clatlon[1])
        )
    plot_ds(invlocfs,'invlocfs',ncols = 3,dims = ['clat','clon'])

    hx0 = invert_filtering(cx,invlocfilters,invflat,invflon)
   
    hmed =  4.16751088375953# np.median(hx0[hx0>0])
    hx0 = hx0/hmed
    print(hmed)
    cx0 = coarse_grain_by_weights(hx0,locfilters,filter_lat,filter_lon)
    x[hx0==0] = 0
    hx = xr.DataArray(
        data = x,
        dims = ['lat','lon'],
        coords = dict(lat = latlon[0],lon= latlon[1])
    )

    hx0 = xr.DataArray(
        data = hx0,
        dims = ['lat','lon'],
        coords = dict(lat = latlon[0],lon= latlon[1])
    )
    cx = xr.DataArray(
        data = cx,
        dims = ['clat','clon'],
        coords = dict(clat = clatlon[0],clon= clatlon[1])
    )
    cx0 = xr.DataArray(
        data = cx0,
        dims = ['clat','clon'],
        coords = dict(clat = clatlon[0],clon= clatlon[1])
    )
    plot_ds({'cx':cx,'cx0':cx0},'lres',ncols = 2,dims = ['clat','clon'])#,cmap = 'inferno')
    plot_ds({'hx':hx,'hx0':hx0,'err':hx - hx0},'hres',ncols = 3,dims = ['lat','lon'])
    
    return 

if __name__=='__main__':
    main()
