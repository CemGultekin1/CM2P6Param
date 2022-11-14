import itertools
import os
import matplotlib.pyplot as plt
from data.load import pass_geo_grid
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
    title_inc = ['sigma','domain','depth','latitude','lsrp']
    title_nam = ['sigma','train-domain','train-depth','latitude','lsrp']
    subplotkwargs = dict(projection=ccrs.PlateCarree(),)
        # projection =  ccrs.PlateCarree()
    # )
    plotkwargs = lambda a : dict(transform=ccrs.PlateCarree(),x = 'geolon' if a else 'lon',y = 'geolat' if a else 'lat',) 
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
        runargs,_ = options(line.split(),key = "run")
        sn = pass_geo_grid(sn,runargs.sigma)
 
        depthvals = sn.depth.values
        timevals = sn.time.values
        targetfolder = os.path.join(target,modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for i,(time,depth) in enumerate(itertools.product(timevals,depthvals)):
            s = sn.sel(time = time,depth = depth)
            s = s.drop('time').drop('depth')
            title_ = f"{title}\ntest-depth: {depth},    time: {time}"
            names = 'u v T'.split() + 'Su Sv ST'.split() + 'Su_true Sv_true ST_true'.split() + 'Su_err Sv_err ST_err'.split()
            ncol = 3
            names = [n if n in list(s.data_vars) else None for n in names ]
            kk = 0
            for namei,name in enumerate(names):
                if name is None:
                    if kk == 0:
                        names[namei] = 'geolon'
                    elif kk==1:
                        names[namei] = 'geolat'
                    kk+=1
            names = np.array(names)
            names = names.reshape([-1,ncol])
            # print(names)
            # names = names[:,[2,0,1]]
            targetfile = os.path.join(targetfolder,f'snapshot_{i}.png')
            nrow = names.shape[0]
            plt.figure(figsize = (60,20))
            
            for ir,ic in itertools.product(range(nrow),range(ncol)):

                ax = plt.subplot(nrow,ncol,ir*ncol + ic + 1,**subplotkwargs)
                if names[ir,ic] is None:
                    continue
                # ax.coastlines()
                # ax.gridlines()
                # print(names[ir,ic],'geo' in names[ir,ic])
                kwargs = plotkwargs(False)
                s[names[ir,ic]].plot(ax = ax,**kwargs)
                
                
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