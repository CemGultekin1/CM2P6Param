import itertools
import os
import matplotlib.pyplot as plt
from plots.metrics import metrics_dataset
from utils.paths import SLURM, R2_PLOTS, EVALS
from utils.xarray import skipna_mean
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
    title_inc = ['sigma','domain','depth','latitude','lsrp']
    title_name = ['sigma','train-domain','train-depth','latitude','lsrp']
    for line in lines:
        print(line)
        modelargs,modelid = options(line.split(),key = "model")
        vals = [modelargs.__getattribute__(key) for key in title_inc]
        vals = [int(val) if isinstance(val,float) else val for val in vals]
        title = ',   '.join([f"{name}: {val}" for name,val in zip(title_name,vals)])
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            continue
        sn = xr.open_dataset(snfile).sel(lat = slice(-85,85))#.isel(depth = [0],co2 = 0).drop(['co2'])
        msn = metrics_dataset(sn,dim = [])
        tmsn = metrics_dataset(sn,dim = ['lat','lon'])

        depthvals = msn.depth.values
        targetfolder = os.path.join(target,modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for i in range(len(depthvals)):
            s = msn.isel(depth = i)
            ts = tmsn.isel(depth = i)

            depthval = depthvals[i]

            title_ = f"{title}\ntest-depth: {depthval}"
            names = "Su Sv ST".split()
            unames = np.unique([n.split('_')[0] for n in list(s.data_vars)])
            names = [n for n in names if n in unames]
            ftypes = ['r2','mse','sc2','corr']
            
            nrows = len(names)
            ncols = len(ftypes)
            _names = np.empty((nrows,ncols),dtype = object)
            for ii,jj in itertools.product(range(nrows),range(ncols)):
                n = f"{names[ii]}_{ftypes[jj]}"
                _names[ii,jj] = n
        
            targetfile = os.path.join(targetfolder,f'maps_{i}.png')

            fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*6,nrows*5))
            for ir,ic in itertools.product(range(nrows),range(ncols)):
                name = _names[ir,ic]
                var = s[name]
                pkwargs = dict()
                if 'r2' in name or 'corr' in name:
                    pkwargs = dict(vmin = 0.5,vmax = 1)
                else:
                    var = np.log10(var)
                var.plot(ax = axs[ir,ic],**pkwargs)
                title = f"{name}:{'{:.2e}'.format(ts[name].values[0])}"
                print(title)
                axs[ir,ic].set_title(title,fontsize=24)
            fig.suptitle(title_,fontsize=24)
            fig.savefig(targetfile)
            flushed_print(title_,'\n\t',targetfile)
            plt.close()
        # return



if __name__=='__main__':
    main()