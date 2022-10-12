import itertools
import os
import sys
import torch
from data.paths import get_filename
from data.load import get_data
from run.train import Timer
from utils.arguments import options
from utils.slurm import flushed_print
import numpy as np
from utils.xarray import concat, unpad
import xarray as xr

import shutil


def save_separate_depths():
    datargs = sys.argv[1:]
    generator,= get_data(datargs,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('all',))
    datargs,_ = options(datargs,key = "data")
    depth = datargs.depth

    timer = Timer()
    timer.start('data')
    for fields,forcings,masks,locations in generator:
        i = locations['itime'][0].item()
        fields = dict(fields,**forcings)
        fields,masks = unpad(fields),unpad(masks)
        fields = concat(fields,masks)
        if depth > 0:
            time = 3 + 5*i + 180
        else:
            time = i + 1 + 180
        fields = fields.expand_dims(dim = {"time": [time]},axis=0)
        fields = fields.expand_dims(dim = {"depth": [depth]},axis=1)
        fields = fields.isel(depth = 0)

        renamekwargs = {i:i.replace('forcings_','S').replace('fields_','') for i in fields.data_vars}
        fields = fields.rename(**renamekwargs)
        fields=fields.chunk(chunks={"time":1,"lon":len(fields.lon),"lat":len(fields.lat)})
        flushed_print(fields)
        filename = get_filename(datargs.sigma,depth,datargs.co2)
        tempfilename = filename.replace('.zarr',f'_temp_{int(depth)}.zarr')
        flushed_print('\t\t\t\t',tempfilename)
        if i==0:
            fields.to_zarr(tempfilename,mode='w')
        else:
            fields.to_zarr(tempfilename,append_dim="time",mode='a')
        timer.end('data')
        flushed_print('\n')
        i+=1
        flushed_print(i,timer)
        timer.start('data')

def concat_zarrs():
    filename = get_filename(4,5,False).split('/')
    root = '/'.join(filename[:-1])
    files = os.listdir(root)
    files = [f for f in files if 'temp' in f]
    sigmas = [int(f.split('_')[1]) for f in files]
    sigmas = np.unique(sigmas)

    hresfile = get_filename(1,5,False)
    hrespath = os.path.join(root,hresfile)
    depthvals = xr.open_zarr(hrespath).st_ocean.values
    for sigma in sigmas:
        if sigma < 16:
            continue
        flushed_print(sigma)
        datasets = []
        _files = [f for f in files if f'coarse_{sigma}_temp' in f]
        for _f in _files:
            path = os.path.join(root,_f)
            ds = xr.open_zarr(path)
            depth = ds.depth.values
            if depth < 1e-3:
                continue
            depth = depthvals[np.argmin(np.abs(depthvals - depth))]
            ds['depth'] = depth
            ds = ds.expand_dims(dim = {"depth": [depth]},axis=1)
            datasets.append(ds)


        path1 = os.path.join(root,get_filename(sigma,5,False))
        print(path1)
        # return
        flushed_print('\tconcatenating')
        ds = xr.concat(datasets,dim = 'depth')

        flushed_print('\twriting')
        ds.to_zarr(path1,mode = "a")
        flushed_print('\twrote')

    files = os.listdir(root)
    files = [f for f in files if 'temp_0.zarr' in f]
    sigmas = [int(f.split('_')[1]) for f in files]
    for sigma in sigmas:
        flushed_print(sigma)
        _files = [f for f in files if f'coarse_{sigma}_' in f]
        path1 = os.path.join(root,get_filename(sigma,0,False))
        if os.path.exists(path1):
            shutil.rmtree(path1)
        for i,_f in enumerate(_files):
            path = os.path.join(root,_f)
            ds = xr.open_zarr(path)
            depth = ds.depth.values
            assert depth <1e-3
            flushed_print(path,path1)
            os.rename(path,path1)
            break

    files = os.listdir(root)
    files = [f for f in files if 'temp_0' in f]


if __name__=='__main__':
    concat_zarrs()
