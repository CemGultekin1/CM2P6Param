import itertools
import os
import matplotlib.pyplot as plt
import xarray as xr
from utils.arguments import options
import numpy as np

def main():
    root = '/scratch/cg3306/climate/CM2P6Param/saves/snapshots/'
    models = '/scratch/cg3306/climate/CM2P6Param/slurm_jobs/deep_eval.txt'
    target = '/scratch/cg3306/climate/CM2P6Param/saves/plots/snapshots/'
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()
    title_inc = ['depth','latitude','linsupres']
    title_nam = ['train-depth','latitude','lsrp']
    # lines = lines[:12]
    for line in lines:
        modelargs,modelid = options(line.split(),key = "model")
        snfile = os.path.join(root,modelid + '.nc')
        title = ',   '.join([f"{name}: {modelargs.__getattribute__(key)}" for key,name in zip(title_inc,title_nam)])
        print(title)
        # continue
        if not os.path.exists(snfile):
            continue
        sn = xr.open_dataset(snfile)
        depthvals = sn.depth.values
        timevals = sn.time.values
        targetfolder = os.path.join(target,modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for i,(time,depth) in enumerate(itertools.product(range(len(timevals)),range(len(depthvals)))):
            s = sn.isel(time = [time],depth = [depth],lon = slice(0,-2)).sel(lon = slice(-180,-120),lat = slice(-60,-20))
            depthval,timeval = depthvals[depth],timevals[time]
            title_ = f"{title}\ntest-depth: {depthval},    time: {timeval}"
            names = np.array(list(s.data_vars))
            names = names.reshape([2,3]).T
            targetfile = os.path.join(targetfolder,f'snapshot_{i}.png')
            fig,axs = plt.subplots(3,3,figsize = (30,30))
            for ir,ic in itertools.product(range(3),range(2)):
                s[names[ir,ic]].plot(ax = axs[ir,ic])
                axs[ir,ic].set_title(names[ir,ic],fontsize=24)

            for ir,ic in itertools.product(range(3),range(2,3)):
                err = s[names[ir,0]] - s[names[ir,1]]
                err.plot(ax = axs[ir,ic])
                axs[ir,ic].set_title(names[ir,0]+'_err',fontsize=24)
            fig.suptitle(title_,fontsize=24)
            fig.savefig(targetfile)
            plt.close()
            if i==len(depthvals):
                break




if __name__=='__main__':
    main()