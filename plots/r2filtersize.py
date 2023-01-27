from argparse import Namespace
import itertools
import os
from re import X
import matplotlib.pyplot as plt
from utils.paths import SLURM, R2_PLOTS, EVALS
import xarray as xr
from utils.arguments import options
from utils.slurm import flushed_print
import numpy as np


def main():
    root = EVALS
    models = os.path.join(SLURM,'trainjob.txt')
    target = R2_PLOTS
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()
    lines = lines[40:] + ['lsrp']
    coords = ['linsupres','seed']
    coordnames = ['latitude_features','CNN_LSRP','training_depth','seed']
    r2vals = None
    for line in lines:
        if line == 'lsrp':
            modelid = 'lsrp'
            coordvals = {}
            for cn,coord in zip(coordnames,coords):
                coordvals[cn] = [0]
            coordvals['LSRP'] = [1]
        else:
            modelargs,modelid = options(line.split(),key = "model")
            coordvals = {}
            for cn,coord in zip(coordnames,coords):
                val = modelargs.__getattribute__(coord)
                if isinstance(val,bool):
                    val = int(val)
                coordvals[cn] = [val]
            coordvals['LSRP'] = [0]
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
            r2 = r2.expand_dims(**coordvals)
            r2.name = key
            r2s.append(r2)
        xr2s = xr.merge(r2s)
        if r2vals is None:
            r2vals = xr2s
        else:
            r2vals = xr.merge([r2vals,xr2s])
    r2vals = r2vals.mean(dim = "seed")
    ylim = [0,1]
    colsep = {'latitude_features':[0,1,0,1],'CNN_LSRP':[0,0,1,1]}
    title_naming = ['latitude','LSRP']
    linsep = 'training_depth'
    xaxisname = 'depth'
    ncol = 4
    rowsep = list(r2vals.data_vars)
    nrow = len(rowsep)
    fig,axs = plt.subplots(nrow,ncol,figsize = (9*ncol,6*nrow))
    for i,j in itertools.product(range(nrow),range(ncol)):
        ax = axs[i,j]
        colsel = {key:val[j] for key,val in colsep.items()}
        rowsel = rowsep[i]
        y = r2vals[rowsel]
        ylsrp = y.sel(LSRP = 1).isel(**{key:0 for key in colsel})
        ylsrp = ylsrp.isel({linsep : 0})
        y = y.sel(**colsel)        
        y = y.sel(LSRP = 0)
        ixaxis = np.arange(len(ylsrp))
        for l in range(len(y[linsep])):
            yl = y.isel({linsep : l})
            ax.plot(ixaxis,yl,label = str(yl[linsep].values))
            ax.plot(ixaxis[l],yl.values[l],'k.',markersize = 12)
        ax.plot(ixaxis,ylsrp,'--',label = 'LSRP')
        ax.set_ylim(ylim)
        ax.set_xticks(ixaxis)
        xaxis = ylsrp[xaxisname].values
        xaxis = ["{:.2e}".format(v) for v in xaxis]
        ax.set_xticklabels(xaxis)
        ax.legend()
        ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.8)
        ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
        if j==0:
            ax.set_ylabel(rowsel)
        if i==0:
            title = ", ".join([f"{n}:{bool(v)}" for n,v in zip(title_naming,colsel.values())])
            ax.set_title(title)
    fig.savefig(os.path.join(target,'depth_comparison.png'))
if __name__=='__main__':
    main()