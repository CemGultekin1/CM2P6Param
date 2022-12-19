import sys
from data.exceptions import RequestDoesntExist
from data.load import get_data
from data.scalars import save_scalar
from utils.arguments import populate_data_options
import xarray as xr
import numpy as np
from utils.xarray import fromtorchdict, make_dimensional, tonumpydict
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
    scalars = None
    multidatargs = populate_data_options(args,non_static_params=['depth','co2'],domain = 'global')
    for datargs in multidatargs:
        try:
            generator, = get_data(datargs,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('train',))
        except RequestDoesntExist:
            print('data not found!')
            generator = None
        if generator is None:
            continue
        mom0 = None
        mom1 = None
        for i,(fields,field_masks,field_coords,info) in enumerate(generator):
            fields = fromtorchdict(fields,field_coords,field_masks,denormalize = True,drop_normalization = True)
            mom0_ = (1 - np.isnan(fields)).sum(dim = ['lat','lon'],skipna = True)
            mom1_ = np.abs(fields).sum(dim = ['lat','lon'],skipna = True)
            mom0 = update_value(mom0,mom0_)
            mom1 = update_value(mom1,mom1_)
            if i==50:
                break
            flushed_print(i)
        
        mean_ = mom1/mom0

        mean_ = add_suffix(mean_,'scale')
        scalars_ = xr.merge([mean_,])
        scalars_ = make_dimensional(scalars_,'depth',info['depth'].item())
        if scalars is None:
            scalars = scalars_
        else:
            scalars= xr.merge([scalars,scalars_])
    print(scalars)
    save_scalar(args,scalars)

if __name__=='__main__':
    main()
