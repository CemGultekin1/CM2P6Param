import itertools
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.paths import SLURM, VIEW_PLOTS, VIEWS
import xarray as xr
from utils.arguments import options
import numpy as np
import cartopy.crs as ccrs

def main():
    root = VIEWS
    models = os.path.join(SLURM,'viewjob.txt')
    target = VIEW_PLOTS
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()
    # lines = ['lsrp'] + lines
    title_inc = ['depth','latitude','lsrp']
    title_nam = ['train-depth','latitude','lsrp']
    subplotkwargs = dict()
        # projection =  ccrs.PlateCarree()
    # )
    plotkwargs = dict()
        # cbar_kwargs = {'shrink':0.6},
    # )
    # ans = 'u v T'.split()
    # ans = ans  + [f"S{a}" for a in ans] + [f"S{a}_true" for a in ans]
    for line in lines:
        if line == 'lsrp':
            modelid = 'lsrp'
            title = 'LSRP'
        else:
            modelargs,modelid = options(line.split(),key = "model")
            title = ',   '.join([f"{name}: {modelargs.__getattribute__(key)}" for key,name in zip(title_inc,title_nam)])
        snfile = os.path.join(root,modelid + '.nc')
        
        if not os.path.exists(snfile):
            continue
        sn = xr.open_dataset(snfile)
        print(sn)
        depthvals = sn.depth.values
        timevals = sn.time.values
        targetfolder = os.path.join(target,modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for i,(time,depth) in enumerate(itertools.product(timevals,depthvals)):
            s = sn.sel(time = time,depth = depth)
            title_ = f"{title}\ntest-depth: {depth},    time: {time}"
            names = list(s.data_vars)
            print(names)
            names = np.array(names)#[n for n in names if n in ans])
            ncol = 3
            if len(names) % ncol != 0:
                df = ncol - (len(names)%ncol)
                names = np.concatenate([names,[None]*df])
            names = names.reshape([ncol,-1]).T
            print(names)
            # names = names[:,[2,0,1]]
            targetfile = os.path.join(targetfolder,f'snapshot_{i}.png')
            def replace_zero_with_nan(x):
                return xr.where(x==0,np.nan,x)
            nrow = names.shape[0]
            plt.figure(figsize = (60,20))
            
            for ir,ic in itertools.product(range(nrow),range(ncol)):

                ax = plt.subplot(nrow,ncol,ir*ncol + ic + 1,**subplotkwargs)

                replace_zero_with_nan(\
                    s[names[ir,ic]]).plot(ax = ax,**plotkwargs)
                
                ax.set_title(names[ir,ic],fontsize=24)

            print(title_)
            plt.suptitle(title_,fontsize=24)
            plt.savefig(targetfile)
            plt.close()
            print(targetfile)
            if i==len(depthvals)-1:
                break



if __name__=='__main__':
    main()