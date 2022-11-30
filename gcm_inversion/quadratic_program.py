
import itertools
import numpy as np
from utils.xarray import plot_ds
import xarray as xr


def pairwise_distance(H,us,grad = False):
    if grad:
        g = np.zeros(us.shape)
    else:
        g = None
    numpts =us.shape[1]*us.shape[2]
    dij = 0
    n = len(us)
    for i,j in itertools.product(range(H.shape[0]),range(H.shape[1])):
        ui = us[i]
        uj = us[j]
        dij += H[i,j]*np.mean(ui*uj)
        g[i] += H[i,j]*uj
    for i,ui in enumerate(us):
        neighborhood = np.arange(n)
        for j in  neighborhood:
            if j == i:
                continue
            uj = us[j]
            
            if grad:
                g[i] += (ui - uj)*2/numpts
    return dij,g


def iterate_over_nonzero_indices(weights,tol = 1e-12):
    rs,cs = np.where(np.abs(weights)>tol)
    for r,c in zip(rs,cs):
        yield r,c

def index2slice(i,j,sigma):
    return [slice(i_*sigma,(i_+5)*sigma + 1,) for i_ in [i,j]]


def check_orthogonality(latf,lonf):
    filters = []
    for latf_,lonf_ in itertools.product(latf,lonf):
        filters.append((latf_.reshape([-1,1]) @lonf_.reshape([1,-1])).reshape(-1))
    filters = np.stack(filters, axis =0 )
    eye = filters @ filters.T
    orterr = np.sum(np.abs(np.diag(np.diag(eye)) - eye))
    if not orterr < 1e-9:
        print('\t\torterr:\t',orterr)
    
def support_size(gs):
    return np.sum(np.abs(gs)>0)
def elementwise_grad_projection(g_,sigma,latf,lonf,weights):
    g = g_.copy()
    supp0 = support_size(g)
    for i,j in iterate_over_nonzero_indices(weights):
        slat,slon = index2slice(i,j,sigma)
        gs = g[slat,slon]
        if np.all(gs == 0):
            continue
        latf_ = latf[i,slat].copy()
        lonf_ = lonf[j,slon].copy()
        latf_ = latf_/np.sqrt(np.sum(np.square(latf_))).reshape([1,-1])
        lonf_ = lonf_/np.sqrt(np.sum(np.square(lonf_))).reshape([1,-1])
        gs = gs - (latf_.T @lonf_) * (latf_@gs@lonf_.T)
        g[slat,slon] = gs
    supp1 = support_size(g)
    # print(supp0,supp1)
    innerproduct = np.sum(g*g_)/np.sqrt(np.sum(np.square(g_)))/np.sqrt(np.sum(np.square(g)))
    if innerproduct < 0:
        print(innerproduct)
        raise Exception
    return g

def opt_loss_grad(H,us,latf,lonf,w,sigma):
    d,gs = pairwise_distance(H,us,grad = True)
    # return d,gs
    for z,(i,j) in enumerate(itertools.product(range(latf.shape[0]),range(lonf.shape[0]))):
        if  np.all(gs[z] == 0):
            continue
        gsz = elementwise_grad_projection(gs[z],sigma,latf[i],lonf[j],w[z])
        gs[z] = gsz
    return d,gs

def optimal_step_size(H,us,gs):
    n = gs.shape[0]
    gg = np.zeros((n,n))
    gu = np.zeros((n,n))
    for i,j in itertools.product(range(n),range(n)):
        gg[i,j] = np.mean(gs[i]*gs[j])
        gu[i,j] = np.mean(gs[i]*us[j])
    Hgg = H*gg
    Hgu = np.sum(H*gu,axis = 1)
    # Q,R = np.linalg.qr(Hgg,mode = 'reduced' )
    u,s,vh = np.linalg.svd(Hgg,full_matrices = False)
    rs = s/s[0]
    I= np.where(rs>1e-3)[0]
    u = u[:,I]
    vh = vh[I,:]
    s = s[I]
    left_inv = u@ np.diag(1/s) @ vh
    topt = left_inv@Hgu
    # topt = np.minimum(topt,0)
    return topt

def plot_iter(us,iternum):
    u = xr.DataArray(
        data = us[len(us)//2],
        dims = ['lat','lon'],
        coords = dict(
            lat = np.arange(us.shape[1]),lon = np.arange(us.shape[2])
        )
    )
    plot_ds({'u':u},f'u_{iternum}',ncols = 1)

def optimize(us,latf,lonf,w,sigma):
    H = get_hessian(us.shape[0])

    for iternum in range(100):
        d,gs = opt_loss_grad(H,us,latf,lonf,w,sigma)
        topt = optimal_step_size(H,us,gs)#np.array([100])#
        tgs = topt.reshape([-1,1,1])*gs
        # print(iternum,np.mean(np.abs(tgs)),np.mean(np.abs(topt)),np.mean(np.abs(gs)))
        us = us - tgs
        if iternum %10 == 0:
            plot_iter(us,iternum)
        print(iternum,d)
    return us

def get_hessian(n,):
    A = np.zeros((n,n))
    for i in range(n):
        neighborhood = range(n)# [(i+s)%n for s in [-1,1]]
        for j in  neighborhood:
            if j==i:
                continue
            A[i,i] +=  1
            A[i,j] -= 1
            A[j,i] -= 1
            A[j,j] +=1
    H = A.T@A
    _,s,_ = np.linalg.svd(H)
    H = H/s[0]
    return H

def main():
    # print(eigenvalues(121,1))
    return

if __name__=='__main__':
    main()
