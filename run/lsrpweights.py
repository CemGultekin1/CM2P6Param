import sys
from data.coords import TIMES
from data.load import get_data, load_xr_dataset
from utils.paths import average_highres_fields_path
from utils.slurm import flushed_print
import torch
import numpy as np
from utils.xarray import  tonumpydict#,fromnumpydict
import xarray as xr
def main():
    depth = 0 if sys.argv[1] == '1' else 5
    args = f'--mode data --depth {depth}'.split()
    ds = load_xr_dataset(args)


    minibatch = None
    params={'batch_size':minibatch,\
        'shuffle':False,\
        'num_workers':8}
    class TorchDatasetWrap(torch.utils.data.Dataset):
        def __init__(self,ds):
            self.ds = ds.copy()
        def __len__(self,):
            return int(np.floor(len(self.ds.time)*TIMES['train'][1]))
        def __getitem__(self,i):
            ds = self.ds.isel(time = i).load()
            ds = ds.drop('time')
            return tonumpydict(ds)

    torchdset = TorchDatasetWrap(ds)
    loader = torch.utils.data.DataLoader(torchdset, **params)
    totds = None
    j = 0
    for data_vars,coords in loader:
        # ds = fromnumpydict(data_vars,coords)
        for key in data_vars:
            dims,val = data_vars[key]
            if isinstance(val,torch.Tensor):
                data_vars[key] = (dims,val.numpy())
        for key in coords:
            if isinstance(coords[key],torch.Tensor):
                coords[key] = coords[key].numpy()
        ds = xr.Dataset(data_vars = data_vars,coords = coords)
        if totds is None:
            totds = ds
        else:
            totds = totds + ds
        j+=1
        flushed_print(j)
    totds = totds/j
    filename = average_highres_fields_path(depth)
    totds.to_netcdf(filename,mode = 'w')

    for sigma in [4,8,12,16]:
        flushed_print(f'sigma:\t{sigma}')
        args = f'--mode data --sigma {sigma} --depth {depth}'.split()
        ds,= get_data(args,torch_flag = False,data_loaders = False)
        ds.save_average_residual_fields()

    
        

   

if __name__=='__main__':
    main()
