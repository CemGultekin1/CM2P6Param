import itertools
import os
import matplotlib.pyplot as plt
from plots.metrics import metrics_dataset
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
    # lines = 'lsrp_0 lsrp_1'.split() + lines
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
        sn = xr.open_dataset(snfile)
        msn = metrics_dataset(sn,reduckwargs = {})
        depthvals = msn.depth.values
        targetfolder = os.path.join(target,modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for i in range(len(depthvals)):
            s = msn.isel(depth = i)
            depthval = depthvals[i]
            title_ = f"{title}\ntest-depth: {depthval}"
            names = "Su Sv ST".split()
            unames = np.unique([n.split('_')[0] for n in list(s.data_vars)])
            names = [n for n in names if n in unames]
            ftypes = ['r2','mse','sc2']
            
            nrows = len(names)
            ncols = len(ftypes)
            _names = np.empty((nrows,ncols),dtype = object)
            for ii,jj in itertools.product(range(nrows),range(ncols)):
                n = f"{names[ii]}_{ftypes[jj]}"
                _names[ii,jj] = n
        
            targetfile = os.path.join(targetfolder,f'maps_{i}.png')

            fig,axs = plt.subplots(nrows,ncols,figsize = (50,30))
            for ir,ic in itertools.product(range(nrows),range(ncols)):
                pkwargs = dict(vmin = 0,vmax = 1)
                var = s[_names[ir,ic]]
                subtitle = _names[ir,ic]
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