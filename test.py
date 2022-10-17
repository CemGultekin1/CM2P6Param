# from utils.arguments import populate_data_options
import sys
from data.coords import REGIONS
from data.gcm_forcing import SingleDomain

from data.load import dataset_arguments, load_xr_dataset
from utils.arguments import options
def main():
    args = sys.argv[1:]
    # pargs = populate_data_options(args,parts = (1,1))
    # print('\n\n'.join(pargs))

    _args,_kwargs = dataset_arguments(args, half_spread = 10)
    sd = SingleDomain(*_args,**_kwargs)
    UF = sd[0]
    import numpy as np
    print(np.any(np.isnan(UF['fields'].lon.values)))
    return
    # prms,_=options(args,key = "data")

    ds_zarr = load_xr_dataset(args)
    sd = SingleDomain(ds_zarr,prms.sigma,boundaries = REGIONS['global'],half_spread = int((10*4)//prms.sigma),)
    

if __name__=='__main__':
    main()
