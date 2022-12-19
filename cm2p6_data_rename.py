import os

root = '/scratch/zanna/data/cm2.6/'

folders = os.listdir(root)
folders = [f for f in folders if '_.zarr' in f]
folders1 = [f.replace('_.zarr','_gcm.zarr') for f in folders]
for f0,f1 in zip(folders,folders1):
    f0 = os.path.join(root,f0)
    f1 = os.path.join(root,f1)
    os.rename(f0,f1)