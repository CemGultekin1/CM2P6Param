import itertools
import os
import matplotlib.pyplot as plt
from models.load import load_old_model
from plots.metrics import metrics_dataset
from utils.paths import TIME_LAPSE_PLOTS, TIME_LAPSE
from utils.xarray import skipna_mean
import xarray as xr
from utils.arguments import options
from utils.slurm import flushed_print
import numpy as np
def main():
    root = TIME_LAPSE
    target = TIME_LAPSE_PLOTS  
    lines = 'G-0 G-1'.split()
    title_inc = ['sigma','domain','depth','latitude','lsrp']
    title_name = ['sigma','train-domain','train-depth','latitude','lsrp']

    for line in lines:


        args = '--filtering gaussian --widths 2 128 64 32 32 32 32 32 4 --kernels 5 5 3 3 3 3 3 3 --batchnorm 1 1 1 1 1 1 1 0'.split()
        args.extend('--mode eval --num_workers 1'.split())
        modelid = line
        
        modelargs,_ = options(args,key = "model")
        vals = [modelargs.__getattribute__(key) for key in title_inc]
        vals = [int(val) if isinstance(val,float) else val for val in vals]
        title = ',   '.join([f"{name}: {val}" for name,val in zip(title_name,vals)])
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            continue
        print(line)
        sn = xr.open_dataset(snfile).isel(co2 = 0,depth = 0)


        # coord_ids = sn.coord_id.values
        targetfolder = os.path.join(target,modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        s = sn

        lats,lons = s.lat.values, s.lon.values
        
        names = "Su Sv ST".split()
        unames = np.unique([n.split('_')[1] for n in list(s.data_vars) if n not in 'lat lon'.split()])
        names = [n for n in names if n in unames]
        
        nrows = len(names)
        ncols = len(lats)
        # _names = np.empty((nrows,1),dtype = object)
        # for ii,jj in itertools.product(range(nrows),range(ncols)):
        #     n = f"{names[ii]}_{ftypes[jj]}"
        #     _names[ii,jj] = n
    
        targetfile = os.path.join(targetfolder,f'time_lapse.png')

        fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*12,nrows*5))
        for ir,ic in itertools.product(range(nrows),range(ncols)):
            var = s.isel(coord_id = ic,time = range(2,302))
            ax = axs[ir,ic]
            # title_ = f"{title}\n coord: ({lat},{lon})"
            true_val = var['true_'+names[ir]].values.reshape([-1])
            pred_val = var['pred_'+names[ir]+'_mean'].values.reshape([-1])
            std_val = var['pred_'+names[ir]+'_std'].values.reshape([-1])
            pred1 = 1.96*std_val + pred_val
            pred_1 = -1.96*std_val + pred_val

            for key,val in dict(true_val = true_val,pred_val = pred_val,std_val = std_val).items():
                print(key,np.any(np.isnan(val)))

            ax.plot(true_val,color = 'tab:blue', label = 'true',linewidth = 2)
            ax.plot(pred_val,color = 'tab:orange', label = 'mean',linewidth = 2)
            ax.plot(pred1,color = 'tab:green', label = '1.96-std',linestyle = 'dotted',alpha = 0.5)
            ax.plot(pred_1,color = 'tab:green', linestyle = 'dotted',alpha = 0.5)#label = '1-std',
            ax.legend()
            if ic == 0:
                ax.set_ylabel(names[ir])
            ax.set_title(f'coord = ({lats[ic]},{lons[ic]})')
        # fig.suptitle(title_,fontsize=24)
        fig.savefig(targetfile)
        flushed_print(targetfile)
        # flushed_print(title_,'\n\t',targetfile)
        plt.close()



if __name__=='__main__':
    main()