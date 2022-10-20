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
    title_name = ['latitude','lsrp','train-depth']
    for line in lines:
        if line == 'lsrp':
            modelid = 'lsrp'
            title = 'LSRP'
        else:
            modelargs,modelid = options(line.split(),key = "model")
            vals = [modelargs.__getattribute__(key) for key in title_inc]
            vals = [int(val) if isinstance(val,float) else val for val in vals]
            title = ',   '.join([f"{name}: {val}" for name,val in zip(title_name,vals)])
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            continue
        sn = xr.open_dataset(snfile)
        for key in 'Su Sv ST'.split():
            sn[f"{key}_r2"] = 1 - sn[f"{key}_mse"]/sn[f"{key}_sc2"]
        depthvals = sn.depth.values
        targetfolder = os.path.join(target,modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for i in range(len(depthvals)):
            s = sn.isel(depth = i)#.isel(lon = slice(0,-1))
            if np.any(np.isnan(s.lon.values)):
                s = s.isel(lon = slice(0,-1))
            depthval = depthvals[i]
            title_ = f"{title}\ntest-depth: {depthval}"
            names = "Su Sv ST".split()
            unames = np.unique([n.split('_')[0] for n in list(s.data_vars)])
            names = [n for n in names if n in unames]
            ftypes = ['r2','mse','sc2']
            
            nrows = len(names)
            ncols = 3
            _names = np.empty((nrows,ncols),dtype = object)
            for ii,jj in itertools.product(range(nrows),range(ncols)):
                n = f"{names[ii]}_{ftypes[jj]}"
                _names[ii,jj] = n
            

            targetfile = os.path.join(targetfolder,f'r2_{i}.png')
            def replace_zero_with_nan(x):
                return xr.where(x==0,np.nan,x)
            fig,axs = plt.subplots(nrows,ncols,figsize = (50,30))
            for ir,ic in itertools.product(range(nrows),range(ncols)):
                if 'r2' in _names[ir,ic]:
                    pkwargs = dict(vmin = 0,vmax = 1)
                    var = s[_names[ir,ic]]
                    subtitle = _names[ir,ic]
                else:
                    pkwargs = dict()
                    var = replace_zero_with_nan(s[_names[ir,ic]])
                    var = np.log10(var)
                    subtitle = f"log10({_names[ir,ic]})"
                var.plot(ax = axs[ir,ic],**pkwargs)
                axs[ir,ic].set_title(subtitle,fontsize=24)
            fig.suptitle(title_,fontsize=24)
            fig.savefig(targetfile)
            flushed_print(title_,'\n\t',targetfile)
            plt.close()
            if i==len(depthvals)-1:
                break



if __name__=='__main__':
    main()