
from data.load import dataset_arguments, get_var_grouping
import torch
from utils.arguments import options
from utils.xarray import fromtensor, fromtorchdict2tensor


# def separate_batch(vec,batchnum):
#     def sub_separate_batch(vec,i,batchnum):
#         if isinstance(vec,torch.Tensor):
#             return vec[batchnum].detach().to('cpu').numpy()
#         if isinstance(vec,dict):
#             return {key:separate_batch(val,batchnum) for key,val in vec.items()}
#         if isinstance(vec,tuple) or isinstance(vec,list):
#             x = [sub_separate_batch(val,i,batchnum) for i,val in enumerate(vec)]
#             x_ = [a for a in x if a is not None]
#             if len(x_)< len(x):
#                 if len(x_) == 0:
#                     print(x)
#                     raise Exception
#                 return x_[0]
#             if isinstance(vec,tuple):
#                 return tuple(x_)
#             else:
#                 return x_
#         if i==batchnum:
#             return vec
#         else:
#             return None
#     return sub_separate_batch(vec,None,batchnum)

# def low_res_dataset():
#     nbatch = 1
#     args = f'--sigma 4 --domain global --depth 5 --minibatch {nbatch} --lsrp 0 --prefetch_factor 1 --num_workers 1 --mode train'.split()
#     from data.load import load_xr_dataset
#     import itertools
#     ds = load_xr_dataset(args)
#     vars = get_var_grouping(args)

#     _args,_kwargs = dataset_arguments(args)
#     # _kwargs['boundaries'] = _kwargs['boundaries'][2]
#     dd = DividedDomain(*_args,**_kwargs)
#     sf  = dd[0]
#     import matplotlib.pyplot as plt
#     fig,axs = plt.subplots(1,3,figsize = (25,10))
#     rows = [
#         'u v T'.split(),
#         'Su Sv ST'.split(),
#         'Su0_res Sv0_res ST0_res'.split()
#     ]
#     ncols = len(rows[0])
#     nrows = len(rows)

#     fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*6,nrows*5))
    
#     for i,j in itertools.product(range(nrows),range(ncols)):
#         if nrows == 1:
#             ax = axs[j]
#         else:
#             ax = axs[i,j]
#         name = rows[i][j]
#         u = sf[name]
#         u.plot(ax = ax)
#         ax.set_title(name)

#     fig.savefig('subgrid_forcings_global.png')
#     plt.close()

def data_loader():
    nbatch = 1
    args = f'--sigma 4 --depth 5 --minibatch {nbatch} --prefetch_factor 1 --num_workers 1 --mode train'.split()
    from data.load import get_data
    ld, = get_data(args,half_spread = 10, torch_flag = True, data_loaders = True,groups = ('all',))
    
    
def average_high_res_data():
    from utils.paths import average_highres_fields_path
    import xarray as xr
    import matplotlib.pyplot as plt
    import itertools
    import numpy as np
    for sigma,isdeep in itertools.product([4,8,12,16],[False,True]):
        filename = average_highres_fields_path(sigma,isdeep)
        data =xr.open_dataset(filename)
        nrows = len(data.data_vars.keys())
        print(sigma,isdeep)
        if isdeep:
            ndepth = len(data.depth)
            for di in range(ndepth):
                fig,axs = plt.subplots(nrows,1,figsize = (6,nrows*5))
                for i,key in enumerate(data.data_vars.keys()):
                    u = np.log10(np.abs(data[key].isel(depth = di)))
                    u.plot(ax = axs[i])
                    axs[i].set_title(key)
                fig.savefig(f'average_highres_fields_{sigma}_{di+1}.png')
                plt.close()
        else:
            fig,axs = plt.subplots(nrows,1,figsize = (6,nrows*5))
            for i,key in enumerate(data.data_vars.keys()):
                u = np.log10(np.abs(data[key]))
                u.plot(ax = axs[i])
                axs[i].set_title(key)

            fig.savefig(f'average_highres_fields_{sigma}_{0}.png')
            plt.close()

def plot_forcings():
    import xarray as xr
    from utils.xarray import  plot_ds
    
    root =  '/scratch/zanna/data/cm2.6/'
    import os
    ds = xr.open_zarr(os.path.join(root,'coarse_4_surface_.zarr')).isel(time = 0,depth = 0)
    plot_ds(ds,'surf.png')
    ds = xr.open_zarr(os.path.join(root,'coarse_4_beneath_surface_.zarr')).isel(time = 0)
    for i in range(len(ds.depth)):
        dsi = ds.isel(depth= i)
        plot_ds(dsi,f'depth_{i}.png')
def scalars():
    nbatch = 1
    args = f'--sigma 4 --depth 5 --domain global --minibatch {nbatch} --prefetch_factor 1 --termperature True --num_workers 1 --mode scalars'.split()
    from data.load import get_data
    from params import SCALAR_PARAMS
    import xarray as xr
    import numpy as np
    from utils.xarray import remove_normalization,fromtorchdict,mask_dataset,plot_ds
    normalizations = SCALAR_PARAMS["normalization"]["choices"]

    generator,= get_data(args,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('train',))
    for fields,field_masks,field_coords,_ in generator:
        fields = fromtorchdict(fields,field_coords,field_masks)
        plot_ds(fields,'scalars_example.png',ncols = 3)
        print(fields)
        return


        

def eval():
    nbatch = 1
    args = f'--sigma 4 --depth 5 --domain four_regions --minibatch {nbatch} --prefetch_factor 1 --lsrp 2 --termperature True --num_workers 1 --mode eval'.split()
    from data.load import get_data
    import xarray as xr
    import numpy as np
    from utils.xarray import fromtorchdict,plot_ds
    stats = None
    def lsrp_pred(respred,tr):
        keys= list(respred.data_vars.keys())
        data_vars = {}
        coords = {key:val for key,val in tr.coords.items()}
        for key in  keys:
            trkey = key.replace('0_res','').replace('1_res','')
            err = tr[trkey] - tr[key]
            data_vars[trkey] = (err.dims,err.values)
            respred[key] = err + respred[key]
            respred = respred.rename({key:trkey})
            tr = tr.drop(key)
        lsrp = xr.Dataset(data_vars =data_vars,coords = coords)
        return (respred,lsrp),tr
    def update_stats(stats,prd,tr,lsrp_flag):
        if lsrp_flag:
            cnn_prd,lsrp_prd = prd
            if stats is None:
                stats = [None,None]
            stats[0] = update_stats(stats[0],cnn_prd,tr,False)
            stats[1] = update_stats(stats[1],lsrp_prd,tr,False)
            return stats

        err = np.square(tr - prd)
        sc2 = np.square(tr)
        err = err.rename({key:f'{key}_mse' for key in err.data_vars})
        sc2 = sc2.rename({key:f'{key}_sc2' for key in sc2.data_vars})
        stats_ = xr.merge([err,sc2])
        if stats is None:
            stats = stats_
        else:
            stats = stats + stats_
        return stats
    runprms,_ = options(args,key = 'run')
    lsrp_flag = runprms.lsrp > 0
    kwargs = dict(contained = '' if not lsrp_flag else 'res')
    generator,= get_data(args,half_spread = 5, torch_flag = False, data_loaders = True,groups = ('test',))
    for fields,forcings,forcing_mask,_,forcing_coords in generator:
        fields_tensor = fromtorchdict2tensor(fields)
        forcings_tensor = fromtorchdict2tensor(forcings,**kwargs)
        outputs = torch.randn(forcings_tensor.shape)
        print(fields_tensor.shape,outputs.shape,forcings_tensor.shape)
        predicted_forcings = fromtensor(outputs,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
        true_forcings = fromtorchdict(forcings,forcing_coords,forcing_mask,denormalize = True)
        if lsrp_flag:
            predicted_forcings,true_forcings = lsrp_pred(predicted_forcings,true_forcings)
        stats = update_stats(stats,predicted_forcings,true_forcings,lsrp_flag)
        plot_ds(stats,'scalars_example.png',ncols = 3)
        return
        


def training():
    nbatch = 1
    args = f'--sigma 12 --depth 0 --minibatch {nbatch} --domain four_regions --prefetch_factor 1 --lsrp 2 --mode train --temperature True --latitude True --num_workers 1'.split()
    from data.load import get_data
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    def compute_stats(forcings,masks):
        nonan = masks.sum(dim = [0,2,3])
        # forcings[masks<1] = 0
        means = forcings.sum(dim = [0,2,3])/nonan
        sc2 = (forcings**2).sum(dim = [0,2,3])/nonan
        std = torch.sqrt(sc2 - means**2)
        forcings[masks<1] = np.nan
        return means.numpy(),std.numpy()
    generator,= get_data(args,half_spread = 3, torch_flag = True, data_loaders = True,groups = ('train',))
    for k,(fields,forcings,masks) in enumerate(generator):
        print(np.any(np.isnan(fields.numpy())),np.any(np.isnan(forcings.numpy())))
        print(compute_stats(fields,(torch.abs(fields)>0).type(torch.float)))
        print(compute_stats(forcings,masks))
        print()
        vecs = []
        for f in [fields,forcings,masks]:
            vecs.extend(torch.unbind(f))
        vecs1 = []
        for f in vecs:
            vecs1.extend(torch.unbind(f))
        n = len(vecs1)
        ncols = 3
        nrows = np.ceil(n/ncols).astype(int)
        fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*6,nrows*5))
        for z,(i,j) in enumerate(itertools.product(range(nrows),range(ncols))):
            ax = axs[i,j]
            if n <= z:
                continue
            ax.imshow(vecs1[z].numpy()[::-1,:])
        fig.savefig(f'training_batch_{k}.png')
        plt.close()
        if k==4:
            break

def main(): 
    training()
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
