import itertools
import os
from data.gcm_forcing import SingleDomain
from data.load import get_high_res_data_location, preprocess_dataset
from utils.arguments import options    
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from utils.slurm import flushed_print
from torch import nn
import torch
def get_wetmasks(sigma,depth):
    args = f'--sigma {sigma} --depth {depth}'.split()
    datargs, _ = options(args,key = "data")
    hres_path = get_high_res_data_location(args)
    print(hres_path)
    hresdata = xr.open_zarr(hres_path,consolidated=False)
    hresdata = preprocess_dataset(args,hresdata)

    sd = SingleDomain(hresdata,datargs.sigma,coarse_grain_needed = True,wetmask = False)#,boundaries=boundaries)
    return sd.get_wet_mask(include_hres=True)
def ecdf(mask,nbins = 16):
    m = mask.values.reshape([-1])
    x = np.linspace(0,1,nbins+1)
    v = np.zeros(nbins)
    for i in range(nbins):
        v[i] = np.sum(m > x[i])/len(m)
    return x[:-1],v


def main():
    root = '/scratch/cg3306/climate/CM2P6Param/saves/plots/wet_masks'
    sigmavals = [4,8,12]
    vf = [21,11,9]
    vf = {s:v for  s,v in zip(sigmavals,vf)}
    depths = [0,5,55,110,330,1497]
    lres_wetmasks = {}
    ecdfs = {}
    for depth,sigma in itertools.product(depths,sigmavals):
        flushed_print(depth,sigma)
        arg = (depth,sigma)
        lres_wetmasks[arg],lres_wetmasks[(depth,1)] = get_wetmasks(sigma,depth)
        ecdfs[arg] = ecdf(lres_wetmasks[arg])
        if (depth,1) not in ecdfs:
            ecdfs[(depth,1)] = ecdf(lres_wetmasks[(depth,1)])
        
    

    select_sigmas = [1,4,8,12,]
    for select_depth in depths:
        fig,axs = plt.subplots(2,2,figsize = (35,15))
        for i in range(len(select_sigmas)):
            ic = i%2
            ir = i//2
            sigma = select_sigmas[i]
            ax = axs[ir,ic]
            if (select_depth,sigma) not in lres_wetmasks:
                continue
            lres_wetmasks[(select_depth,sigma)].plot(ax = ax)
            ax.set_title(f'sigma = {sigma},depth = {select_depth}')
        fig.savefig(os.path.join(root,f'soft_wet_mask_{select_depth}.png'))
        plt.close()

    
    fig,axs = plt.subplots(2,2,figsize = (35,15))
    for i in range(4):
        ic = i%2
        ir = i//2
        sigma = select_sigmas[i]
        ax = axs[ir,ic]
        for depth in depths:
            if (depth,sigma) not in ecdfs:
                continue
            ax.plot(*ecdfs[(depth,sigma)],label = f'{depth}m')
        ax.legend()
    fig.savefig(os.path.join(root,'wet_density.png'))
    plt.close()

    for depth,sigma in itertools.product(depths,sigmavals):
        arg = (depth,sigma)
        if arg not in lres_wetmasks:
            continue
        wm = lres_wetmasks[arg]
        pool = nn.MaxPool2d(vf[sigma],stride = 1)
        trshs = [0.5,0.75,0.8,0.9,0.95,1]
        fig,axs = plt.subplots(2,3,figsize = (35,20))
        
        
        for i,trsh in enumerate(trshs):
            wmv = wm.values.copy()
            m = wmv>=trsh-1e-3
            wmv[m] = 1
            wmv[~m] = 0
            shp = np.array(wmv.shape) - (vf[sigma] - 1)
            wmv = torch.from_numpy(wmv.reshape(1,1,wmv.shape[0],wmv.shape[1]))
            with torch.no_grad():
                pwmv = 1 - pool(1 - wmv)

            pwmv = pwmv.numpy().reshape(*shp)
            n = np.sum(pwmv)/shp[0]/shp[1]
            sp = (vf[sigma] - 1)//2
            pwmv = xr.DataArray(
                data = pwmv,
                dims = ["lat","lon"],
                coords = dict(
                    lat = wm.lat.values[sp:-sp],
                    lon = wm.lon.values[sp:-sp],
                )
            )
            ic = i%3
            ir = i//3
            ax = axs[ir,ic]
            pwmv.plot(ax = ax)
            ax.set_title(f'wet density >= {trsh}, ratio = {n}')
            print(depth,sigma,trsh,n)
        fig.savefig(os.path.join(root,f'forcing_values_{sigma}_{depth}.png'))
        plt.close()
   
    
if __name__ == '__main__':
    main()