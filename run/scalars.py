import sys
from data.load import get_data
from data.scalars import save_scalar
import xarray as xr
import numpy as np
from utils.xarray import fromtorchdict
from utils.slurm import flushed_print

def update_value(a,b):
    if a is None:
        return b
    return a+b
def add_suffix(ds,suffix):
    for key in list(ds.data_vars):
        newname = f'{key}_{suffix}'
        # if newname in ds.data_vars.keys():
        #     ds = ds.drop(newname)
        ds = ds.rename({key:newname})
    return ds

def main():
    args = sys.argv[1:]
    generator,= get_data(args,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('train',))
    mom0 = None
    mom1 = None
    # mom2 = None
    i= 0
    for i,(fields,field_masks,field_coords,_) in enumerate(generator):
        fields = fromtorchdict(fields,field_coords,field_masks,denormalize = True,drop_normalization = True)
        mom0_ = (1 - np.isnan(fields)).sum(dim = ['lat','lon'],skipna = True)
        mom1_ = np.abs(fields).sum(dim = ['lat','lon'],skipna = True)
        # mom2_ = np.square(fields).sum(dim = ['lat','lon'],skipna = True)
        mom0 = update_value(mom0,mom0_)
        mom1 = update_value(mom1,mom1_)
        if i==50:
            break
        flushed_print(i)
        # break
    
    mean_ = mom1/mom0
    # std_ = np.sqrt(mom2/mom0 - mean_**2)
    # mean_ = xr.where(np.isnan(mean_),0,mean_)
    # std_ = xr.where(np.isnan(std_),1,std_)

    mean_ = add_suffix(mean_,'scale')
    # std_ = add_suffix(std_,'std')
    # scalars = xr.merge([mean_,std_])
    scalars = xr.merge([mean_,])
    print(scalars)
    save_scalar(args,scalars)

if __name__=='__main__':
    main()
