import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import itertools
from models.nets.lsrp_build import forward_difference

from models.nets.lsrp import ConvolutionalLSRP

def test_model(sigma,save_true_force = False):
    root = '/scratch/cg3306/climate/saves/lsrp/'
    path = os.path.join(root,f'inv_weights_{sigma}.nc')
    filters = xr.open_dataset(path)
    def coarse_grain(u,selkwargs):
        u = u.values
        u[u!=u]= 0
        ubar = (filters.forward_lat.values @ u) @ filters.forward_lon.values.T
        ubar = xr.DataArray(
            data = ubar,
            dims = ["clat","clon"],
            coords = dict(
                clat = (["clat"],filters.clat.data),
                clon  = (["clon"],filters.clon.data)
            )
        )
        return ubar.sel(selkwargs)

    
    def save_true_forcing():
        from data.load import load_xr_dataset
        args = f'--sigma {sigma} --mode data'.split()
        ds = load_xr_dataset(args).isel(time = 0)
        selkwargs = {'clon':slice(-180,-90),'clat':slice(-40,40)}
        print('computing true forcing')
        dvdx = forward_difference(ds.v,"ulon")
        dvdy = forward_difference(ds.v,"ulat")
        S1 = coarse_grain(ds.u*dvdx,selkwargs) + coarse_grain(ds.v*dvdy,selkwargs)
        S0 = coarse_grain(ds.u,selkwargs) *  forward_difference(coarse_grain(ds.v,selkwargs),"clon") \
            + \
            coarse_grain(ds.v,selkwargs) *  forward_difference(coarse_grain(ds.v,selkwargs),"clat")
        Strue = S0 - S1
        ubar = coarse_grain(ds.u,selkwargs)
        vbar = coarse_grain(ds.v,selkwargs)

        Strue = Strue.to_dataset(name = "true_data")
        Strue = Strue.assign(ubar = ubar,vbar = vbar)
        root = '/scratch/cg3306/climate/saves/lsrp/'
        path = os.path.join(root,f'true_forcing_example_{sigma}.nc')
        Strue.to_netcdf(path = path)
        # return ubar,vbar,Strue,clat,clon
    def get_true_forcing():
        root = '/scratch/cg3306/climate/saves/lsrp/'
        path = os.path.join(root,f'true_forcing_example_{sigma}.nc')
        return  xr.open_dataset(path)
    if save_true_force:
        save_true_forcing()
        return 
    Strue = get_true_forcing()
    # print(Strue)
    clat,clon = Strue.clat.values,Strue.clon.values
    ubar = Strue.ubar
    vbar = Strue.vbar
    lsr_forcing = ConvolutionalLSRP(sigma,clat)
    S = lsr_forcing.forward(ubar,vbar,vbar)
    fig,axs = plt.subplots(1,2,figsize = (30,10))
    Strue.true_data.sel(clon = slice(*S.lon.values[[0,-1]]),clat = slice(*S.lat.values[[0,-1]])).plot(ax = axs[0])
    S.plot(ax = axs[1])
    axs[1].set_title('Convolutional LSRP Output')
    axs[0].set_title('True Forcing')
    fig.savefig(f'/scratch/cg3306/climate/saves/plots/lsrp/lsrp_test_{sigma}.png')

def plot_weights(sigma):
    span = 20
    lsrpm = ConvolutionalLSRP(sigma,np.arange(span*2+1)-span)
    w0,w1 = lsrpm.single_latitude_weights(0)
    # w0,w1 = w0[2:-2,2:-2,2:-2,2:-2],w1[2:-2,2:-2,2:-2,2:-2]
    def get2x2(w):
        shp = list(w.shape)
        nx = shp[0]*shp[2]
        ny = shp[1]*shp[3]
        W = np.zeros((nx,ny))
        for i,j in itertools.product(range(shp[0]),range(shp[1])):
            subw  = w[i,j]
            ii = slice(i*shp[2],(i+1)*shp[2])
            jj = slice(j*shp[3],(j+1)*shp[3])
            W[ii,jj] = subw
            W[ii.stop-1,:] = np.nan
            W[:,jj.stop-1] = np.nan
        
        return xr.DataArray(W)


    W0,W1 = get2x2(w0),get2x2(w1)
    fig,axs = plt.subplots(1,2,figsize = (50,20))
    W0.plot(ax = axs[0])
    W1.plot(ax = axs[1])
    specs = dict(fontsize = 30)
    fig.suptitle('$S_T = \overline{v} \overline{\partial_y} \overline{T} - \overline{ v \partial_y T } + \overline{u} \overline{\partial_x} \overline{T} - \overline{ u \partial_x T }$',**specs)
    axs[0].set_title('$\overline{v}  \overline{\partial_y} \overline{T} - \overline{ v \partial_y T }$',**specs)
    axs[1].set_title('$\overline{u}  \overline{\partial_x} \overline{T} - \overline{ u \partial_x T }$ ',**specs)
    for i in range(2):
        axs[i].set_ylabel('latitude',**specs)
        axs[i].set_xlabel('longitude',**specs)
    fig.savefig(f'/scratch/cg3306/climate/saves/plots/lsrp/lsrp_weights_{sigma}.png')

            

def main():
    
    # return
    for sigma in range(4,6,4):
        print('sigma = ',sigma)
        # test_model(sigma,save_true_force= True)
        test_model(sigma,save_true_force= False)
        # plot_weights(sigma)
    

if __name__=='__main__':
    main()