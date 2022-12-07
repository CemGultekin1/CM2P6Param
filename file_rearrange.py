path = '/scratch/zanna/data/cm2.6/'
import xarray as xr
import os
files = os.listdir(path)
new_files = [f for f in files if '__.zarr' in f]
for i in range(len(new_files)):
    print(new_files[i])
    ds = xr.open_dataset(os.path.join(path,new_files[i]))
    print(new_files[i])
    print(ds)
new_names = [f.replace('__.zarr','_.zarr') for f in new_files]

# import shutil
# shutil.rmtree('dir_path')
