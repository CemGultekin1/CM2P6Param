import itertools
import json
import os
from data.coords import REGIONS
from data.gcm_dataset import MultiDomainGCMDataset, MDMTorchOutput
from data.load import load_dataset, load_xr_dataset
import matplotlib.pyplot as plt
import numpy as np
from utils.arguments import options
import xarray as xr
def main(): 
    


    return
    root = '/scratch/zanna/data/cm2.6/'
    file = 'coarse-3D-data-sigma-14.zarr'
    path = os.path.join(root,file)
    ds = xr.open_zarr(path)
    print(ds)
    return 
    sigmas = [4,8,12,16]
    for sigma in sigmas:
        args = f"--sigma {sigma} --domain four_regions --depth 5 --parts 3 3 --temperature True --latitude True --normalization standard".split()

        scalarns,scalarid = options(args,key = "scalar")
        print(sigma,'\t',scalarid)
        file = '/scratch/cg3306/climate/saves/scalars.json'
        if os.path.exists(file):
            with open(file) as f:
                scalars = json.load(f)
            if scalarid in scalars:
                for key,val in scalars[scalarid].items():
                    print('\t',key,val)
                continue
        print('scalars are not found!')


    return
    # dd = load_dataset(args,torch = True)
    # # print(dd.compute_scalars(time_window = 2))
    # dd.set_additional_spread(10*sigma)
    # n = 9
    # fig,axs = plt.subplots(n,2,figsize = (15,n*5))
    # for i in range(n):
    #     inputs,outputs = dd[i]
    #     axs[i,0].imshow(inputs[0])
    #     axs[i,1].imshow(outputs[0])
    #     print(i,inputs.shape,outputs.shape)
    # fig.savefig('dummy.png')
    # return 

    
    ds = load_xr_dataset(args)
    # sigma = 4
    # dd = TorchDatasetWrap(ds,sigma,boundaries = REGIONS["four_regions"],parts = (1,1),linsupres = True,lsrp_span = 5)
    # nrows,ncols = dd.parts[0]
    # inputs,outputs = dd[0]
    # print(inputs.shape,outputs.shape)
    # dd.set_additional_spread(10*sigma)
    # inputs,outputs = dd[0]
    # print(inputs.shape,outputs.shape)
    # dd.set_additional_spread(20*sigma)
    # inputs,outputs = dd[0]
    # print(inputs.shape,outputs.shape)


    dd = MultiDomainGCMDataset(ds,sigma,boundaries = REGIONS["global"],parts =(3,3))#linsupres = False,lsrp_span = 8,
    nrows,ncols = 3,3
    figaxs = {}
    for key_ in 'u v T'.split():
        figaxs[key_] = {}
        figaxs[key_][0] = plt.subplots(nrows,ncols,figsize = (5*ncols,5*nrows))
        figaxs[key_][1] = plt.subplots(nrows,ncols,figsize = (5*ncols,5*nrows))
        # figaxs[key_][2] = plt.subplots(nrows,ncols,figsize = (5*ncols,5*nrows))
    
    for i,j in itertools.product(range(ncols),range(nrows)):
        l = i*nrows + j
        UF = dd[l]
        for r,(inout,val) in enumerate(UF.items()):
            print(inout,j,i,r)
            for field,fgx in figaxs.items():
                fig,axs = fgx[r]
                ax = axs[j,i]
                val[field].plot(ax = ax)
                ax.set_title(f"{inout}-{field} ({j},{i})")

    for key in figaxs.keys():
        for i,(fig,axs) in enumerate(figaxs[key].values()):
            fig.savefig(f"dummy_{key}_{i}.png")
    plt.close()
    

if __name__=='__main__':
    main()