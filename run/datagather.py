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
    arg = int(sys.argv[1]) - 1

    path = '/scratch/cg3306/climate/CM2P6Param/jobs/datagen.txt'
    with open(path) as f:
        ls = f.readlines()

    ls = [l.strip() for l in ls]
    upper_limit = (arg+1)*NSEC#40
    lower_limit = arg*NSEC #30
    flushed_print(lower_limit,upper_limit)
    for i in range(lower_limit,upper_limit):
        datargs = ls[i].split()
        path1 = get_preliminary_low_res_data_location(datargs)
        path0 = get_low_res_data_location(datargs).replace('.zarr','_.zarr')
        overwrite =  not os.path.exists(path0)
        append_zarr(path0,path1,overwrite)
                

        




if __name__=='__main__':
    run()
