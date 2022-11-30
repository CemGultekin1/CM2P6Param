import numpy as np
import xarray as xr

def tonumpydict(x:xr.Dataset):
    vars = list(x.data_vars)
    data_vars = {}
    for var in vars:
        data_vars[var] = (list(x[var].dims),x[var].values)

    coords = {}
    for c in list(x.coords):
        coords[c] = x[c].values
        if c == 'time':
            coords[c] = np.array(coords[c]).astype(str)
    return data_vars,coords

    
def plot_ds(ds,imname,ncols = 3,dims = ['lat','lon'],cmap = 'seismic'):
    kwargs = dict(dims = dims,cmap = cmap)
    if isinstance(ds,list):
        for i,ds_ in enumerate(ds):
            imname_ = imname.replace('.png',f'-{i}.png')
            plot_ds(ds_,imname_,ncols = ncols,**kwargs)
        return

    if isinstance(ds,dict):
        for name,var in ds.items():
            var.name = name
        plot_ds(xr.merge(list(ds.values())),imname,ncols = ncols,**kwargs)
        return
    import matplotlib.pyplot as plt
    import matplotlib
    import itertools
    
    
    excdims = []
    for key in ds.data_vars.keys():
        u = ds[key]
        dim = list(u.dims)
        excdims.extend(dim)
    excdims = np.unique(excdims).tolist()
    for d in dims:
        if d not in excdims:
            raise Exception
        excdims.pop(excdims.index(d))

    flat_vars = {}
    for key in ds.data_vars.keys():
        u = ds[key]
        eds = [d for d in u.dims if d in excdims if len(ds.coords[d])>1]
        base_sel = {d : 0 for d in u.dims if d in excdims if len(ds.coords[d])==1}
        neds = [len(ds.coords[d]) for d in eds]
        inds = [range(nd) for nd in neds]
        for multi_index in itertools.product(*inds):
            secseldict = {ed:mi for ed,mi in zip(eds,multi_index)}
            seldict = dict(base_sel,**secseldict)
            keyname = key + '_'.join([f"{sk}_{si}" for sk,si in secseldict.items()])
            flat_vars[keyname] = u.isel(**seldict)
    vars = list(flat_vars.keys())
    nrows = int(np.ceil(len(vars)/ncols))
    fig,axs = plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*5))
    print('nrows,ncols\t',nrows,ncols)
    for z,(i,j) in enumerate(itertools.product(range(nrows),range(ncols))):
        if nrows == 1 and ncols == 1:
            ax = axs
        elif nrows == 1:
            ax = axs[j]
        elif ncols == 1:
            ax = axs[i]
        else:
            ax = axs[i,j]
        if z >= len(vars):
            continue
        u = flat_vars[vars[z]]
        cmap = matplotlib.cm.get_cmap(cmap)
        cmap.set_bad('black',.4)
        u.plot(ax = ax,cmap = cmap)
        ax.set_title(vars[z])
    fig.savefig(imname)
    plt.close()



def concat(**kwargs):
    data_vars = dict()
    coords = dict()
    for name,var in kwargs.items():
        data_vars[name] = (var.dims,var.data)
        coords_ = {key:coo for key,coo in var.coords.items() if key in var.dims}
        coords = dict(coords,**coords_)
    return xr.Dataset(data_vars = data_vars,coords = coords)
