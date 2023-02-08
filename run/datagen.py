import sys
from data.paths import get_low_res_data_location, get_preliminary_low_res_data_location
from data.load import get_data
from run.train import Timer
from utils.arguments import options
from utils.slurm import flushed_print
from utils.xarray import plot_ds
# from utils.xarray import plot_ds
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
    datargs,_ = options(datargs,key = "data") # 333
    initflag = False
    dst = None
    time = Timer()
    time.start('data')
    for data_vars,coords in generator:
        time.end('data')
        data_vars,coords = torch2numpy(data_vars,coords)
        ds = xr.Dataset(data_vars = data_vars,coords = coords)
        if dst is not None:
            if ds.time.values[0] != dst.time.values[0]:
                flushed_print(dst.time.values[0],time)
                chk = {k:len(dst[k]) for k in list(dst.coords)}
                if not initflag:
                    dst = dst.chunk(chunks=chk)
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
        time.start('data')
    if dst is not None:
        flushed_print(dst.time.values[0],time)
        chk = {k:len(dst[k]) for k in list(dst.coords)}
        dst = drop_timeless(dst)
        dst = dst.chunk(chunks=chk)
        dst.to_zarr(filename,mode='a',append_dim = 'time')
if __name__=='__main__':
    run()
