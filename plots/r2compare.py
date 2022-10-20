from argparse import Namespace
import itertools
import os
import matplotlib.pyplot as plt
from utils.paths import SLURM, R2_PLOTS, EVALS
import xarray as xr
from utils.arguments import options
from utils.slurm import flushed_print
import numpy as np


def main():
    root = EVALS
    models = os.path.join(SLURM,'evaljob.txt')
    target = R2_PLOTS
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()
    lines = ['lsrp'] + lines
    title_inc = ['latitude','linsupres','depth']
    r2vals = None
    for line in lines:
        if line == 'lsrp':
            modelid = 'lsrp'
        else:
            modelargs,modelid = options(line.split(),key = "model")
            vals = [modelargs.__getattribute__(key) for key in title_inc]
            vals = [int(val) if isinstance(val,float) else val for val in vals]
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            continue
        sn = xr.open_dataset(snfile)
        r2s = []
        for key in 'Su Sv ST'.split():
            mse = sn[f"{key}_mse"]
            sc2 = sn[f"{key}_sc2"]
            sc2 = sc2.fillna(0)
            mse = mse.fillna(0)
            r2 = 1 - mse.sum(dim = ["lat","lon"])/sc2.sum(dim = ["lat","lon"])
            r2 = r2.expand_dims({'train_depth': [modelargs.depth]})
            r2.name = key
            r2s.append(r2)
        r2 = xr.merge(r2s)
        if r2vals is None:
            r2vals = r2
        else:
            r2vals = xr.merge([r2vals,r2])
    print(r2vals)

            



if __name__=='__main__':
    main()