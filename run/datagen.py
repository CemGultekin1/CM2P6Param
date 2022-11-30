import sys
from data.paths import get_preliminary_low_res_data_location
from data.load import get_data
from utils.arguments import options
from utils.slurm import flushed_print
import xarray as xr
import torch
def torch2numpy(data_vars,coords):
    for key in data_vars:
        dims,val = data_vars[key]
        if isinstance(val,torch.Tensor):
            data_vars[key] = (dims,val.numpy())
    for key in coords:
        if isinstance(coords[key],torch.Tensor):
            coords[key] = coords[key].numpy()
    return data_vars,coords
def drop_timeless(ds:xr.Dataset):
    timeless_params = []
    for key in ds.data_vars.keys():
        if 'time' not in ds[key].dims:
            timeless_params.append(key)
    for tp in timeless_params:
        ds = ds.drop(tp)
    return ds

def run():
    datargs = sys.argv[1:]
    generator,= get_data(datargs,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('all',))
    filename = get_preliminary_low_res_data_location(datargs)
    datargs,_ = options(datargs,key = "data")
    initflag = False
    dst = None
    for data_vars,coords in generator:
        # print(data_vars,coords)
        # return
        data_vars,coords = torch2numpy(data_vars,coords)
        ds = xr.Dataset(data_vars = data_vars,coords = coords)
        if dst is not None:
            if ds.time.values[0] != dst.time.values[0]:
                flushed_print(ds.time.values[0])
                chk = {k:len(dst[k]) for k in list(dst.coords)}

                if not initflag:
                    dst = dst.chunk(chunks=chk)
                    # print(dst)
                    # return
                    dst.to_zarr(filename,mode='w')
                    initflag = True
                else:
                    dst = drop_timeless(dst)
                    dst = dst.chunk(chunks=chk)
                    dst.to_zarr(filename,mode='a',append_dim = 'time')
                dst = None
        if dst is None:
            dst = ds
        else:
            dst = xr.merge([dst,ds])

if __name__=='__main__':
    run()
