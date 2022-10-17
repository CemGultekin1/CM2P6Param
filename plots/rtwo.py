import itertools
import os
import matplotlib.pyplot as plt
from utils.paths import SLURM, R2_PLOTS, EVALS
import xarray as xr
from utils.arguments import options
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
    title_nam = ['latitude','lsrp','train-depth']
    for line in lines:
        if line == 'lsrp':
            modelid = 'lsrp'
            title = 'LSRP'
        else:
            modelargs,modelid = options(line.split(),key = "model")
            title = ',   '.join([f"{name}: {int(modelargs.__getattribute__(key))}" for key,name in zip(title_inc,title_nam)])
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            continue
        sn = xr.open_dataset(snfile)
        depthvals = sn.depth.values
        targetfolder = os.path.join(target,modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for i in range(len(depthvals)):
            s = sn.isel(depth = i).isel(lon = slice(0,-1))
            depthval = depthvals[i]
            title_ = f"{title}\ntest-depth: {depthval}"
            names = list(s.data_vars)
            names = np.unique([n.split('_')[0] for n in names])
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
                if ic==0:
                    s[_names[ir,ic]] = 1-s[_names[ir,ic]]
                replace_zero_with_nan(s[_names[ir,ic]]).plot(ax = axs[ir,ic])
                axs[ir,ic].set_title(_names[ir,ic],fontsize=24)

            print(title_)
            fig.suptitle(title_,fontsize=24)
            fig.savefig(targetfile)
            plt.close()
            if i==len(depthvals)-1:
                break



if __name__=='__main__':
    main()