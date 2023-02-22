import itertools
from data.load import load_xr_dataset
from data.paths import get_low_res_data_location
import numpy as np
def get_dataset(filtering):
    args = f'--filtering {filtering} --sigma 4'.split()
    ds_zarr,_ = load_xr_dataset(args)
    data_address = get_low_res_data_location(args)
    return ds_zarr,data_address
def check_existing_datasets():
    dsgauss,address1 = get_dataset('gaussian')
    dsgcm,address2  = get_dataset('gcm')
    dsgauss,dsgcm = [x.isel(time = 0,lat = range(100,200),lon = range(100,200)) for x in [dsgauss,dsgcm ]]
    print(address1,address2)
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(3,1)
    dsgauss.u.plot(ax = axs[0])
    dsgcm.u.plot(ax = axs[1])
    dvar = np.log10(np.abs(dsgcm.u - dsgauss.u))
    dvar.plot(ax = axs[2])
    plt.savefig('different_datasets.png')
def check_forcings():
    root = '/scratch/zanna/data/cm2.6/coarse_datasets/'
    import xarray as xr
    select = dict(time = 0,depth =0)#,lat = range(100,500),lon = range(100,500))
    dss = {i:xr.open_zarr(root + f'coarse_4_surface_gaussian_{i}_1.zarr').isel(**select) for i in [0,0]}
    fields = [f'S{x}{y}' for x in 'u v temp'.split() for y in ['','_res']] + 'u v temp'.split()
    k0,k1 = list(dss.keys())
    import matplotlib.pyplot as plt
    for f in fields:
        fig,axs = plt.subplots(2,3,figsize = (27,9))
        for ir in range(2):
            if ir==0:
                v0 = np.log10(np.abs(dss[k0][f]))
                v1 = np.log10(np.abs(dss[k1][f]))
            else:
                v0 = np.abs(dss[k0][f])
                v1 = np.abs(dss[k1][f])
            adv = v0 - v1
            
            vmax = np.amax([np.amax((x if ir==0 else np.abs(x)).fillna(-np.inf).values) for x in [v0,v1]])
            vmin = np.amin([np.amin((x if ir==0 else -np.abs(x)).fillna(np.inf).values) for x in [v0,v1]])
            
            v0.plot(ax = axs[ir,0],vmax = vmax,vmin=vmin,cmap = 'RdBu_r')
            v1.plot(ax = axs[ir,1],vmax = vmax,vmin=vmin,cmap = 'RdBu_r')

            n0 = np.sqrt(np.nanmean(v0**2))
            n1 = np.sqrt(np.nanmean(v1**2))
            n2 = np.nanmean(adv)
            adv.plot(ax = axs[ir,2])

            axs[ir,0].set_title(n0)
            axs[ir,1].set_title(n1)
            axs[ir,2].set_title(n2)
        print(f + '.png')
        plt.savefig(f + '.png')
        plt.close()
check_forcings()