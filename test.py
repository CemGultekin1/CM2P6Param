
from data.load import dataset_arguments, get_var_grouping
from data.low_res import DividedDomain
import torch


def separate_batch(vec,batchnum):
    def sub_separate_batch(vec,i,batchnum):
        if isinstance(vec,torch.Tensor):
            return vec[batchnum].detach().to('cpu').numpy()
        if isinstance(vec,dict):
            return {key:separate_batch(val,batchnum) for key,val in vec.items()}
        if isinstance(vec,tuple) or isinstance(vec,list):
            x = [sub_separate_batch(val,i,batchnum) for i,val in enumerate(vec)]
            x_ = [a for a in x if a is not None]
            if len(x_)< len(x):
                if len(x_) == 0:
                    print(x)
                    raise Exception
                return x_[0]
            if isinstance(vec,tuple):
                return tuple(x_)
            else:
                return x_
        if i==batchnum:
            return vec
        else:
            return None
    return sub_separate_batch(vec,None,batchnum)

def low_res_dataset():
    nbatch = 1
    args = f'--sigma 4 --domain four_regions --depth 5 --minibatch {nbatch} --prefetch_factor 1 --num_workers 1 --mode train'.split()
    from data.load import load_xr_dataset
    import itertools
    ds = load_xr_dataset(args)
    vars = get_var_grouping(args)

    _args,_kwargs = dataset_arguments(args)
    _kwargs['boundaries'] = _kwargs['boundaries'][2]
    dd = DividedDomain(*_args,**_kwargs)
    sf  = dd[0]
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(1,3,figsize = (25,10))
    rows = [
        'u v T'.split(),
        'Su Sv ST'.split(),
        'lsrp_res_Su lsrp_res_Sv lsrp_res_ST'.split()
    ]
    ncols = len(rows[0])
    nrows = len(rows)

    fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*6,nrows*5))
    
    for i,j in itertools.product(range(nrows),range(ncols)):
        if nrows == 1:
            ax = axs[j]
        else:
            ax = axs[i,j]
        name = rows[i][j]
        u = sf[name]
        u.plot(ax = ax)
        ax.set_title(name)

    fig.savefig('subgrid_forcings_global.png')
    plt.close()

def data_loader():
    nbatch = 1
    args = f'--sigma 4 --depth 5 --minibatch {nbatch} --prefetch_factor 1 --num_workers 1 --mode train'.split()
    from data.load import get_data
    ld, = get_data(args,half_spread = 10, torch_flag = True, data_loaders = True,groups = ('all',))
    
    

def main(): 
    low_res_dataset()
    return
    import xarray as xr
    import matplotlib.pyplot as plt
    import itertools
    import numpy as np

    sf = xr.open_dataset('subgrid_forcings.nc')
    for z in range(3):
        fields = 'u v T'.split()
        lsrpfields = ['lsrp_'+f for f  in fields]
        forcings = ['S'+f for f  in fields]
        lsrpforcings = ['lsrp_'+f for f  in forcings]
        rows = [
            fields,
            lsrpfields,
            forcings,
            lsrpforcings,
            ['ugrid_wetmask','ugrid_wetmask','tgrid_wetmask']
        ]
            
        ncols = len(rows[0])
        nrows = len(rows)
        if z<2:
            fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*6,nrows*5))
        else:
            fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*18,nrows*5))
        for i,j in itertools.product(range(nrows),range(ncols)):
            if nrows == 1:
                ax = axs[j]
            else:
                ax = axs[i,j]
            name = rows[i][j]
            u = sf[name]
            # print(name,np.any(np.isnan(u.values)))
            if z > 0:
                dims = u.dims
                grid =  'u' if np.any(['u' in d for d in dims]) else 't'
                lon = grid + 'lon'
                lat = grid + 'lat'
                if z==1:
                    u = u.sel({lat: slice(-30,30),lon:slice(-130,-70)})
                else:
                    u = u.sel({lat: slice(60,90)})
            u.plot(ax = ax)
            ax.set_title(name)
        if z == 0 :
            fig.savefig('subgrid_forcings_global.png')
        else:
            fig.savefig(f'subgrid_forcings_local_{z}.png')
        plt.close()
    return 
    nbatch = 2
    args = f'--sigma 4 --depth 5 --minibatch {nbatch} --prefetch_factor 1 --num_workers 1'.split()
    from data.load import get_data
    from params import SCALAR_PARAMS
    import xarray as xr
    import numpy as np
    normalizations = SCALAR_PARAMS["normalization"]["choices"]

    generator,= get_data(args,half_spread = 2, torch_flag = False, data_loaders = True,groups = ('train',))
    tot_scalars = {norm : {} for norm in normalizations}
    time_limit = 64
    for fields,forcings,masks,coords in generator:
        fields = {key:tuple(val) for key,val in fields.items()}
        forcings = {key:tuple(val) for key,val in forcings.items()}
        masks = {key:tuple(val) for key,val in masks.items()}
        for t in range(nbatch):
            print('\n'*6)
            ifield = separate_batch(fields,t)
            iforcing = separate_batch(forcings,t)
            imask = separate_batch(masks,t)
            icoord = separate_batch(coords,t)
            for key,coor in icoord.items():
                icoord[key] = np.asarray(coor)
                if not np.issubdtype(icoord[key].dtype, np.number):
                    icoord[key] = icoord[key].astype(object)
            # for f,name in zip([ifield,iforcing,imask,],['fields','forcings','masks']):
            #     for key in f:
            #         dims,vals = f[key]
            #         print(f'{t}\t{name}\t',key,dims,vals.shape)
            # for key in icoord:
            #     print(f'{t}\tcoords\t',key,icoord[key])
            # print(ifield)
            # print(icoord)
            ds = xr.Dataset(\
                data_vars = dict(ifield,**iforcing,**imask),\
                    coords = icoord)
            print(ds)
        break
    return
    datargs = '--sigma 4'.split()
    from data.load import load_xr_dataset
    ds = load_xr_dataset(datargs)
    from data.gcm_forcing import SingleDomain
    sdm = SingleDomain(ds,4,half_spread = 5,var_grouping = [('u v T'.split()),('Su Sv ST lsrp_res_Su lsrp_res_Sv lsrp_res_ST'.split())])
    UF = sdm[0]
    print(UF)
    return
    datargs = '--sigma 4'.split()
    from data.paths import get_filename
    from utils.arguments import options
    datargs,_ = options(datargs,key = "data")
    filename = get_filename(datargs.sigma,datargs.depth >1e-3,False)
    import xarray as xr
    ds = xr.open_zarr(filename)
    ds = ds.isel(time = 0,depth = 0)
    print(ds)
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(3,4,figsize= (30,30))
    for i,key in enumerate('u v T'.split()):
        ds[key].plot(ax = axs[i,0])
        ds[f'S{key}'].plot(ax = axs[i,1])
        ds[f'lsrp_S{key}'].plot(ax = axs[i,2])
        err = ds[f'S{key}'] - ds[f'lsrp_S{key}']
        err.plot(ax = axs[i,3])
    fig.savefig('coarse_fig.png')

    return
    

if __name__=='__main__':
    main()
