from data.paths import get_filename,get_high_res_grid_location
from intake import open_catalog
import xarray as xr
import numpy as np
def hres_grid():
    cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/GFDL_CM2.6.yaml")
    ds = cat["GFDL_CM2_6_grid"].to_dask()
    # ds = xr.open_dataset(get_high_res_grid_location())
    coords = list(ds.coords)
    actual_coords = []
    for coord in coords:
        actual_coords.extend(ds[coord].dims)
    actual_coords = list(np.unique(np.array(actual_coords)))
    actual_vars = [c for c in coords if c not in actual_coords]
    data_vars = {}
    for var in actual_vars:
        data_vars[var] = (ds[var].dims,ds[var].values)
    _coords = {c: ds[c].values for c in actual_coords}
    ds_ = xr.Dataset(data_vars = data_vars,coords= _coords)
    ds_.to_netcdf(get_high_res_grid_location(),mode = 'w')
    # ds_.to_netcdf(get_high_res_grid_location().replace('.nc','_.nc'),mode = 'w')
    return
def main():
    path = get_filename(1,1e3,True)
    path = path.replace('.zarr','_.zarr')
    print(path)
    cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/GFDL_CM2.6.yaml")
    ds = cat["GFDL_CM2_6_one_percent_ocean_3D"].to_dask()
    ds = ds.drop('salt')
    dsi = ds.isel(time = [0,])
    print(dsi)
    print(path)
    
    dsi.to_zarr(path,mode='w')

if __name__=='__main__':
    main()