import os
import sys
from data.paths import get_filename
from data.load import get_data
from utils.arguments import options
from utils.slurm import flushed_print
import numpy as np
from utils.xarray import concat, fromnumpydict, unpad
import xarray as xr

import shutil


def run():
    datargs = sys.argv[1:]
    generator,= get_data(datargs,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('all',))
    datargs,_ = options(datargs,key = "data")


    filename = get_filename(datargs.sigma,datargs.depth >1e-3,False)
    dst = None
    initflag = False
    for data_vars,coords in generator:
        ds = fromnumpydict(data_vars,coords)
        flushed_print(ds.time.values[0],ds.depth.values)
        if dst is not None:
            if ds.time.values[0] != dst.time.values[0]:
                chk = {k:len(dst[k]) for k in list(dst.coords)}
                dst = dst.chunk(chunks=chk)
                # dst['time'] = np.array(dst['time'].values,dtype = object)
                if not initflag:
                    dst.to_zarr(filename,mode='w')
                    initflag = True
                else:
                    dst.to_zarr(filename,mode='a',append_dim = 'time')
                dst = None
        if dst is None:
            dst = ds
        else:
            dst = xr.merge([dst,ds])

    return
    for _ in range(3):
        return
        # for key in data_vars:
        #     print([[k,v] for k,v in key.items()])
        # print([type(key) for key in data_vars])
        # return
        fields = dict(fields,**forcings)
        fields,masks = unpad(fields),unpad(masks)
        fields = concat(fields,masks)
        
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
    run()
