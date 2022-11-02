from tkinter import E
import xarray as xr 
path = '/scratch/ag7531/mlruns/19/bae994ef5b694fc49981a0ade317bf07/artifacts/forcing/'
ds = xr.open_zarr(path)
print(ds)