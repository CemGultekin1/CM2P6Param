from scipy.ndimage import gaussian_filter
import numpy as np
from utils.xarray import plot_ds
import xarray as xr
sigma = 4
def coarse_grain(x):
    fx = gaussian_filter(x,sigma = sigma,mode = 'wrap')
    sfx = np.zeros(fx.shape)
    ndims = len(fx.shape)
    for ax in range(ndims):
        for i in range(sigma):
            sfx += np.roll(fx, -i-1, axis = ax)
    sfx = sfx/(sigma ** ndims)
    for ax in range(ndims):
        sfx = sfx.take(indices = np.arange(sfx.shape[ax]//sigma)*sigma,axis= ax)
    return  sfx

def cut_nan_axis(x):
    ndims = len(x.shape)
    for ax in range(ndims):
        nanmask = np.isnan(x)
        axis = list(range(ndims))
        axis.pop(ax)
        axial_mask = ~np.any(nanmask,axis = tuple(axis))
        print(axial_mask)
        x = x.take(indices = np.arange(x.shape[ax])[axial_mask],axis = ax)
    return x


def cg_weights(s):
    n = sigma*s#(s+6)
    edgecut = slice(3,-3)
    x = np.eye(n)
    cxs = []
    for i in range(n):
        cx_ = coarse_grain(x[:,i])#[edgecut]
        cx_ = np.stack([cx_],axis = len(cx_.shape))
        cxs.append(cx_)
    cx = np.concatenate(cxs,axis = 1)

    q,r = np.linalg.qr(cx.T)
    return cx,q@np.linalg.inv(r).T

def demo():
    cg1,icg1 = cg_weights(24)
    cg2,icg2 = cg_weights(48)
    n1 = cg1.shape[1]
    n2 = cg2.shape[1]
    print(cg1.shape)
    print(cg2.shape)

    x  = np.cos(np.linspace(0,1,n2)*4*np.pi)
    xbar = cg2@x
    xbar1 = cg1@x[:n1]

    x0 = icg2 @ xbar
    x1 = icg1 @ xbar1
    edgecut = slice(0,n1)#slice(3*sigma,n1 - 3*sigma)
    x0h = x0[edgecut]
    x1h = x1[edgecut]
    xh = x[edgecut]
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(3,1)
    axs[0].plot(xh)
    axs[1].plot(x0h)
    axs[2].plot(x1h)
    fig.savefig('cg_plots.png')