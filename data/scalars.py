import xarray as xr
import os
from utils.arguments import options
from utils.paths import SCALARS

def get_scalar_path(args):
    _,scalarid = options(args,key = 'scalar')
    file = os.path.join(SCALARS,scalarid)
    return file
def load_scalars(args,):
    path = get_scalar_path(args)
    if os.path.exists(path):
        return xr.load_dataset(path).load()
    return None
def save_scalar(args,ds):
    ds.to_netcdf(get_scalar_path(args))
