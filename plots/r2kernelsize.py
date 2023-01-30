import itertools
import os
import matplotlib.pyplot as plt
from utils.paths import  all_eval_path
from utils.xarray import drop_unused_coords, skipna_mean
import xarray as xr
import numpy as np

def class_functions(Foo):
    return [func for func in dir(Foo) if callable(getattr(Foo, func)) and not func.startswith("__")]

MODELCODES = {
    'model': {'fcnn':'','lsrp:0':'lsrp','lsrp:1':'lsrp1'},
    'domain': {'four_regions':'R4','global':'G'},
    'latitude':{1:'L',0:''},
    'temperature':{1:'T',0:''},
    'lsrp':{0:'',1:'-lsr',2:'-lsr1'},
}
namecut = dict(
    model =  ['lsrp:0','lsrp:1'],
)
def build_name(ckeys,cvals):
    name = ''
    for mk in MODELCODES:
        if mk not in ckeys:
            continue
        subf = cvals[ckeys.index(mk)]
        nd = MODELCODES[mk][subf]
        if mk in namecut:
            if subf in namecut[mk]:
                return  name + nd
        else:
            name =  name + nd
    return name


def append_names(stats):
    coords = stats.coords
    cvals = [coords[key].values for key in coords.keys()]

    ckeys = list(coords.keys())
    mkeys = list(MODELCODES.keys())
    mkeys = [mk for mk in mkeys if mk in ckeys]
    names = []
    for valp in itertools.product(*cvals):
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

    stats['name'] = names
    return stats
    modelsdict = []
    for uname in uniqnames:
        _stats = xr.where(names == uname,stats,np.nan)
        _nancount= xr.where(np.isnan(_stats),0,1)
        _values = xr.where(np.isnan(_stats),0,_stats)
        st = _values.sum(dim = mkeys)/_nancount.sum(dim = mkeys)
        st = st.expand_dims(dim = {'modelname':[uname]})
        modelsdict.append(st.copy())
    # models = xr.merge(modelsdict)
    # models = xr.where(models.training_depth == models.depth,models,np.nan)
    # models = models.isel(training_depth = 0,sigma = 3,co2 = 0,depth = 0).drop(['co2','depth','training_depth'])
    # print(models.ST_r2)

def basiplot(stats):
    # stats = stats.isel(training_depth = 0,depth = 0, co2 = 0,lsrp = [0,1],model = [0,1]).drop(['training_depth','depth','co2'])
    stats = stats.isel(lsrp = [0,1],model = [0,1])#.drop(['training_depth','depth','co2'])
    # print(stats)
    # sc2 = stats.isel(sigma = 0).Su_sc2.values.reshape([-1])
    # names = stats.isel(sigma = 0).name.values.reshape([-1])
    # for i in range(len(sc2)):
    #     print(names[i],sc2[i])
    # return
    cnames ={k:0 for k in stats.coords.keys()}
    dropcoords = ['training_depth','depth','co2','sigma']
    for dc in dropcoords:
        cnames.pop(dc)
    cnames = list(cnames.keys())
    unames = np.unique(stats.name.values)
    stats_ns = []
    for un in unames:
        stats_n = xr.where(stats.name == un,stats,np.nan).drop('name')
        stats_n = skipna_mean(stats_n,dim = cnames)
        stats_n = stats_n.expand_dims(dim = {'name' : [un]},axis= 0)
        stats_ns.append(stats_n)
    stats_ns = xr.merge(stats_ns).isel(co2 = 0,training_depth = 0,depth = 0).drop(['training_depth','depth','co2'])
    keepnames = np.mean(np.isnan(stats_ns.Su_r2.values),axis = 1)  < 1
    keepids = np.arange(len(keepnames))[keepnames]
    stats_ns = stats_ns.isel(name = keepids)
    # print(stats_ns.name.values)
    namesort = np.argsort(skipna_mean(stats_ns, dim = 'sigma').Su_r2.values)
    
    ranks = {}
    for name in stats_ns.name.values:
        ranks[name] = 0
        if 'R4' in name:
            ranks[name] -= 1e6
        if 'lsr' in name:
            ranks[name] += 1e4
        if 'lsrp' in name:
            ranks[name] += 1e5
        ranks[name] +=  len(name.replace('R4','').replace('G',''))
    namesort = np.argsort(list(ranks.values()))



    varnames = list(stats_ns.data_vars.keys())
    vartypes = np.unique([vn.split('_')[1] for vn in varnames])
    colors = 'r b g k'.split()
    markers = 'o ^ v < >'.split()
    for vtype in vartypes:
        vnselect = [vn for vn in varnames if vn.split('_')[1] == vtype]
        ncols = len(vnselect)
        nrows = 1
        fig,axs = plt.subplots(nrows,ncols, figsize = (ncols*5,nrows*5))
        for i in range(ncols):
            vname = vnselect[i]
            ax = axs[i]
            for j in range(len(stats_ns.sigma)):
                vals = stats_ns.isel(sigma = j)
                y = vals[vname].values[namesort]
                x = np.arange(len(y))
                xticklabels =  [str(x) for x in stats_ns.name.values[namesort].tolist()]
                if vtype in ['r2','corr']:
                    ax.plot(x,y,\
                        label = f"\u03C3 = {vals.sigma.values.item()}",\
                        color = colors[j],\
                        marker = markers[j],linestyle = 'None')
                else:
                    ax.semilogy(x,y,\
                        label = f"\u03C3 = {vals.sigma.values.item()}",\
                        color = colors[j],\
                        marker = markers[j],linestyle = 'None')
                ax.set_xticks(x)
                ax.set_xticklabels(xticklabels,rotation=45)
            if vtype in ['r2','corr']:
                ax.set_ylim([0,1.01])
            ax.grid( which='major', color='k', linestyle='--',alpha = 0.5)
            ax.grid( which='minor', color='k', linestyle='--',alpha = 0.5)
            if vtype in ['r2','corr']:
                ax.legend(loc = 'lower right')
            else:
                ax.legend(loc = 'upper right')
            ax.set_title(vname)
            ax.set_ylabel(vtype)
        fig.savefig(f"{vtype}.png")
        plt.close()


def group_by_extension(ds,groups):
    grouping = {}
    for name in ds.data_vars.keys():
        for kc in groups:
            if kc in name:
                if kc not in grouping:
                    grouping[kc] = []
                v = ds[name]
                v.name = v.name.replace(f"_{kc}",'')
                grouping[kc].append(v)
                break

    for kc,vals in grouping.items():
        grouping[kc] = xr.merge(vals)
    return list(grouping.values())

def ax_sel_data(dsl,i,j):
    ds = dsl[j]
    keys = list(ds.data_vars.keys())
    keysi = keys[i] 
    return ds[keysi],keysi
def get_lsrp_values(stats):
    return drop_unused_coords(stats.isel(seed = 0,latitude = 0, domain = 0, temperature = 0,lsrp = 0,training_depth = 0, kernel_size = 7, depth = 0, co2 = 0,model = 1))

def main():
    stats = xr.open_dataset(all_eval_path())
    # print(stats)
    lsrp_ = get_lsrp_values(stats).isel(sigma= range(4))

    # print(stats)
    # return
    stats_ = stats.isel(seed = 0,latitude = 1, domain = 1, temperature = 1, sigma = range(4),co2= 0,training_depth = 0,depth = 0)
    

    
    for r2corr in range(2):#itertools.product(range(3),range(2)):
        # sigma = stats_.sigma.values[sigma_i]
        stats = stats_.copy()#isel(sigma = sigma_i)
        lsrp = lsrp_.copy()#isel(sigma = sigma_i)

        stats = group_by_extension(stats,'r2 corr'.split())[r2corr]
        lsrp = group_by_extension(lsrp,'r2 corr'.split())[r2corr]
        r2corr_str = 'r2' if not r2corr else 'corr'
        fcnn = stats.isel(model = 0)
        fcnn_lsrp = fcnn.isel(lsrp = 1)
        fcnn = fcnn.isel(lsrp = 0)

        fcnn,fcnn_lsrp,lsrp = drop_unused_coords(fcnn),drop_unused_coords(fcnn_lsrp),drop_unused_coords(lsrp)

        ylim = [0,1]
        nrows = 3
        ncols = 4
        fig,axs = plt.subplots(nrows,ncols,figsize = (9*ncols,6*nrows))

        fcnn_by_sigma = [fcnn.isel(sigma = ss) for ss in range(4)]
        fcnn_lsrp_by_sigma = [fcnn_lsrp.isel(sigma = ss) for ss in range(4)]
        lsrp_by_sigma = [lsrp.isel(sigma = ss) for ss in range(4)]

        for i,j in itertools.product(range(nrows),range(ncols)):
            ax = axs[i,j]
            yfcnn,rowsel = ax_sel_data(fcnn_by_sigma,i,j)
            yfcnn_lsrp,_ = ax_sel_data(fcnn_lsrp_by_sigma,i,j)
            ylsrp,_ = ax_sel_data(lsrp_by_sigma,i,j)
            ixaxis = np.arange(len(yfcnn.kernel_size))
            markers = 'o v ^ <'.split()
            colors = [f'tab:{x}' for x in 'blue orange green red'.split()]
            
            ax.plot(ixaxis,yfcnn,\
                color = colors[0], marker = markers[0],label = f'FCNN',linestyle = 'None')
            ax.plot(ixaxis,yfcnn_lsrp,\
                color = colors[1], marker = markers[1],label = f'FCNN+LSRP',linestyle = 'None')
            ax.axhline(y = ylsrp.values.item(),color = colors[3], label = f'LSRP')

            ax.set_ylim(ylim)
            ax.set_xticks(ixaxis)
            xaxis = yfcnn.kernel_size.values
            xaxis = [str(v) for v in xaxis]
            ax.set_xticklabels(xaxis)
            ax.legend()
            ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.8)
            ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
            if j==0:
                ax.set_ylabel(rowsel)
            if i==0:
                title =f'\u03C3 = {stats.sigma.values[j]}'
                title = r2corr_str.upper() + ', ' + title
                ax.set_title(title)
            ax.set_xlabel('kernel_size')
        if not os.path.exists('saves/plots/kernel_size_comparison/'):
            os.makedirs('saves/plots/kernel_size_comparison/')
        
        fig.savefig(f'saves/plots/kernel_size_comparison/{r2corr_str}.png')
        print(f'saves/plots/kernel_size_comparison/{r2corr_str}.png')



    
if __name__=='__main__':
    main()