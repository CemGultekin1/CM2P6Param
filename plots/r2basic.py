import itertools
import os
import matplotlib.pyplot as plt
from utils.paths import  all_eval_path
import xarray as xr
import numpy as np

def class_functions(Foo):
    return [func for func in dir(Foo) if callable(getattr(Foo, func)) and not func.startswith("__")]

MODELCODES = {
    'model': {'fcnn':'','lsrp:0':'lsrp__','lsrp:1':'lsrp1__'},
    'training_domain': {'global':'G','four_regions':'R4'},
    'latitude_features':{1:'L',0:''},
    'temperature':{1:'T',0:''},
    'lsrp':{0:'',1:'-lsr',2:'-lsr1'},
}

def build_name(ckeys,cvals):
    name = ''
    for mk in MODELCODES:
        nd = MODELCODES[mk][cvals[ckeys.index(mk)]]
        if '__' in nd:
            return  name + nd.replace('__','')
        else:
            name =  name + nd
    return name


def read(stats):
    # data_vars = stats.data_vars
    
    coords = stats.coords
    cvals = [coords[key].values for key in coords.keys()]
    ckeys = list(coords.keys())
    mkeys = list(MODELCODES.keys())
    names = []
    
    for valp in itertools.product(*cvals):
        # print(ckeys,valp)
        name = build_name(ckeys,valp)
        names.append(name)
    
    uniqnames = np.unique(names)
    shp = [len(cv) for cv in cvals]
    names = np.reshape(names,shp)
    
    for name, datavar in stats.data_vars.items():
        dims = datavar.dims
        break
    
    dimsi = [ckeys.index(dm) for dm in dims]
    names = np.transpose(names,dimsi)
    names = xr.DataArray(dims = dims, data = names,coords = stats.coords)
    for uname in uniqnames:
        _stats = xr.where(names == uname,stats,np.nan)
        _nancount= xr.where(np.isnan(_stats),0,1)
        _values = xr.where(np.isnan(_stats),0,_stats)
        st = _values.sum(dim = mkeys)/_nancount.sum(dim = mkeys)
        print(uname,st,)
        # return 



def main():
    stats = xr.open_dataset(all_eval_path())
    statsmin = stats.min(dim = 'seed',skipna = True)
    statsmax = stats.max(dim = 'seed',skipna = True)
    read(statsmax)


    # print(statsmax)
    return
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