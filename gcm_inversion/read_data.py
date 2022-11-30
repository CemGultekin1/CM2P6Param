from data.load import load_xr_dataset
from transforms.grids import get_grid_vars, logitudinal_expansion, logitudinal_expansion_dataset
import xarray as xr
import numpy as np
def get_grid(depth,longitude_expand = 0):
    ds,_ = load_xr_dataset(f'--mode data --depth {depth}'.split())
    ds = ds.isel(time = 0)
    grid,_ = get_grid_vars(ds)
    
    # u= ds.u.load()
    # u = u.rename(ulat = "lat",ulon = "lon")
    # u.name = 'u'
    # u = logitudinal_expansion(u,3*sigma +1)
    grid = logitudinal_expansion_dataset(grid,longitude_expand)
    return grid
