# from utils.arguments import populate_data_options
import sys
from data.coords import REGIONS
from data.gcm_dataset import MultiDomain, MultiDomainDataset
from data.gcm_forcing import SingleDomain
from data.gcm_lsrp import DividedDomain
import torch
from data.load import dataset_arguments, get_data, load_dataset, load_xr_dataset
from utils.arguments import options
def main():
    args = sys.argv[1:]
    runargs,_ = options(args,key = "run")
    assert runargs.mode == "view"
    ds, = load_dataset(args,half_spread = 5, torch_flag = False, data_loaders = True,groups = ('test',))
    fields,forcings,field_masks,forcing_masks,info=ds[0]
    # test_generator, = get_data(args,half_spread = 5, torch_flag = False, data_loaders = True,groups = ('test',))
    # for fields,forcings,field_masks,forcing_masks,info in test_generator:
    #     break
    print(fields['u']['lon'])
    return 

    _args,_kwargs = dataset_arguments(args, half_spread = 5,torch_flag = False)
    dd = SingleDomain(*_args,**_kwargs)
    print(dd.final_boundaries['lat'],dd.final_boundaries['lon'])
    print(dd.final_local_lres_coords[0][[0,-1]],dd.final_local_lres_coords[1][[0,-1]])
    UF = dd[0]
    print(UF['fields'].lat.values[[0,-1]],UF['fields'].lon.values[[0,-1]])
    print(dd.shape)
    # fields,forcings,field_mask,forcing_mask,info = UF
    # print(fields.keys())
    # print(fields['u'].keys())
    # print(torch.any(torch.isnan(fields['u']['lon'])))
    return
    # prms,_=options(args,key = "data")

    ds_zarr = load_xr_dataset(args)
    sd = SingleDomain(ds_zarr,prms.sigma,boundaries = REGIONS['global'],half_spread = int((10*4)//prms.sigma),)
    

if __name__=='__main__':
    main()
