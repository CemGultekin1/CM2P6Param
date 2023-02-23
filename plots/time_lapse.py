import itertools
import os
import matplotlib.pyplot as plt
from models.load import load_old_model
from plots.metrics import metrics_dataset
from utils.paths import SLURM, TIME_LAPSE_PLOTS, TIME_LAPSE
from utils.xarray import skipna_mean
import xarray as xr
from utils.arguments import options
from utils.slurm import flushed_print
import numpy as np
def main():
    root = TIME_LAPSE
    target = TIME_LAPSE_PLOTS  
    #lines = ['G-0']
    
    models = os.path.join(SLURM,'trainjob.txt')
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()

    lines = [lines[i] for i in [0,2]]

    title_inc = ['sigma','domain','depth','latitude','lsrp']
    title_name = ['sigma','train-domain','train-depth','latitude','lsrp']

    for line in lines:


        args = line.split()
        args.extend('--mode eval --num_workers 1'.split())
        # modelid = line
        
        modelargs,modelid = options(args,key = "model")
        vals = [modelargs.__getattribute__(key) for key in title_inc]
        vals = [int(val) if isinstance(val,float) else val for val in vals]
        title = ',   '.join([f"{name}: {val}" for name,val in zip(title_name,vals)])
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            continue
        print(line)

        sn = xr.open_dataset(snfile).isel(co2 = 0,depth = 0)

        evalfile = snfile.replace('/time_lapse','/evals')
        evalexists =  os.path.exists(evalfile)
        if evalexists:
            evsn = xr.open_dataset(evalfile).isel(co2 = 0,depth = 0)


        # coord_ids = sn.coord_id.values
        targetfolder = os.path.join(target,modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        s = sn

        lats,lons = s.lat.values, s.lon.values
        
        names = "Su Sv ST".split()
        names = names[:1]
        unames = np.unique([n.split('_')[1] for n in list(s.data_vars) if n not in 'lat lon'.split()])
        names = [n for n in names if n in unames]
        
        nrows = len(lats)
        ncols = 1
        targetfile = os.path.join(targetfolder,f'time_lapse.png')

        fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*12,nrows*5))
        # print(ncols,nrows)
        for i,(ir,ic) in enumerate(itertools.product(range(nrows),range(ncols))):
            var = s.isel(coord_id = ir,time = range(4,304))
            ax = axs[ir]
            # title_ = f"{title}\n coord: ({lat},{lon})"
            name = names[0]
            if evalexists:
                t2 = evsn[name+'_true_mom2']
                p2 = evsn[name+'_pred_mom2']
                c2 = evsn[name+'_cross']
                r2 = 1 - (t2 - 2*c2 + p2)/t2
                r2 = r2.sel(lat = var.lat.values.item(),lon = var.lon.values.item(),method = 'nearest').values.item()

            

            true_val = var['true_'+name].values.reshape([-1])
            pred_val = var['pred_'+name+'_mean'].values.reshape([-1])
            std_val = var['pred_'+name+'_std'].values.reshape([-1])
            pred1 = 1.96*std_val + pred_val
            pred_1 = -1.96*std_val + pred_val

            for key,val in dict(true_val = true_val,pred_val = pred_val,std_val = std_val).items():
                print(key,'is any nan values:\t ',np.any(np.isnan(val)))

            ax.plot(true_val,color = 'tab:blue', label = 'true',linewidth = 2)
            ax.plot(pred_val,color = 'tab:orange', label = 'mean',linewidth = 2)
            ax.plot(pred1,color = 'tab:green', label = '1.96-std',linestyle = 'dotted',alpha = 0.5)
            ax.plot(pred_1,color = 'tab:green', linestyle = 'dotted',alpha = 0.5)#label = '1-std',
            ax.legend()
            if ic == 0:
                ax.set_ylabel(name)
            if evalexists:
                ax.set_title(f'coord = ({lats[ir]},{lons[ir]}), r2 = {r2}')
            else:
                ax.set_title(f'coord = ({lats[ir]},{lons[ir]})')
        # fig.suptitle(title_,fontsize=24)
        fig.savefig(targetfile)
        flushed_print(targetfile)
        # flushed_print(title_,'\n\t',targetfile)
        plt.close()



if __name__=='__main__':
    main()