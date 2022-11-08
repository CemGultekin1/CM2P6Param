import itertools
import os
import sys
from data.paths import get_low_res_data_location, get_preliminary_low_res_data_location
from data.load import get_data
from utils.arguments import options
from utils.slurm import flushed_print
import xarray as xr
import torch

NSEC= 10
def append_zarr(path0,path1,overwrite):
    if not os.path.exists(path1):
        print(path1.split('/')[-1])
        return 
    ds = xr.open_zarr(path1)
    for i in range(len(ds.time.values)):
        dst = ds.isel(time = [i]).load()
        if i==0 and overwrite:
            dst.to_zarr(path0,mode = 'w')
        else:
            dst.to_zarr(path0,mode = 'a',append_dim = 'time')
        flushed_print(path0.split('/')[-1],path1.split('/')[-1],i)
def run():
    for sigma,depth,co2 in itertools.product([4,8,12,16],[0,5],[False,True]):
        path0 = None
        for isec in range(NSEC):
            datargs = f'--sigma {sigma} --depth {depth} --co2 {co2} --section {isec} {NSEC}'.split()
            path1 = get_preliminary_low_res_data_location(datargs)
            path0 = get_low_res_data_location(datargs).replace('.zarr','_.zarr')
            append_zarr(path0,path1,isec == 0)
                

        




if __name__=='__main__':
    run()
