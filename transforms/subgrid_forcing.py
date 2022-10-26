from typing import Callable
from transforms.coarse_grain_inversion import coarse_grain_projection
from transforms.grids import forward_difference, get_grid_vars, ugrid2tgrid
import xarray as xr
from utils.xarray import concat



def hreslres(u,v,T,coarse_grain_u,coarse_grain_t,projections = None):
    '''
    converts hres :u,v,t: into lres versions by coarse-graining
    including their derivatives across latitude and longitude
    '''
    ugrid,tgrid = get_grid_vars(u),get_grid_vars(T)
    u_t,v_t = ugrid2tgrid(u,v,ugrid,tgrid)
    if projections is not None:
        import matplotlib.pyplot as plt
        def plotsave(u_t,u_t1,name):
            fig,axs = plt.subplots(1,2,figsize = (25,10))
            u_t.plot(ax = axs[0])
            u_t1.plot(ax = axs[1])
            fig.savefig(f'{name}.png')
            plt.close()
        u_t1 = coarse_grain_projection(u_t,projections,prefix = 't')
        v_t1 = coarse_grain_projection(v_t,projections,prefix = 't')
        T1 = coarse_grain_projection(T,projections,prefix = 't')
        u1 = coarse_grain_projection(u,projections,prefix = 'u')
        v1 = coarse_grain_projection(v,projections,prefix = 'u')
        plotsave(u,u1,'u')
        plotsave(v,v1,'v')
        plotsave(T,T1,'T')
        plotsave(u_t,u_t1,'u_t')
        plotsave(v_t,v_t1,'v_t')
        u_t = u_t1
        v_t = v_t1
        T = T1
        u = u1
        v = v1
        

    uvars = dict(u=u,v=v)
    tvars = dict(u=u_t,v=v_t, T = T)
    def subhres_lres(hresdict,grid,cg):
        lres = {x:cg(y) for x,y in hresdict.items()}
        for val in lres.values():
            gridbar = get_grid_vars(val)
            break
        dlat = {f"d{x}dlat":forward_difference(y,grid,"lat") for x,y in hresdict.items()}
        dlon = {f"d{x}dlon":forward_difference(y,grid,"lon") for x,y in hresdict.items()}
        hres = dict(hresdict,**dlat,**dlon)
        dlat = {f"d{x}dlat":forward_difference(y,gridbar,"lat") for x,y in lres.items()}
        dlon = {f"d{x}dlon":forward_difference(y,gridbar,"lon") for x,y in lres.items()}
        lres = dict(lres,**dlat,**dlon)
        return hres,lres

    uhres,ulres = subhres_lres(uvars,ugrid,coarse_grain_u)
    thres,tlres = subhres_lres(tvars,tgrid,coarse_grain_t)
    return uhres,ulres,thres,tlres

def _subgrid_forcing(hres,lres,key,coarse_grain,):
    
    adv1 = hres["u"]*hres[f"d{key}dlon"] + hres["v"]*hres[f"d{key}dlat"]
    adv1 = coarse_grain(adv1)
    
    adv2 = lres["u"]*lres[f"d{key}dlon"] + lres["v"]*lres[f"d{key}dlat"]
   
    return  adv2 - adv1
def subgrid_forcing(u:xr.DataArray,v:xr.DataArray,T:xr.DataArray,coarse_grain_u:Callable,coarse_grain_t:Callable,**kwargs):
    '''
    :u,v,T: high resolution variables U-grid and T-grid
    :coarse_grain_u,coarse_grain_t: coarse graining methods for U-grid and T-grid separately
    '''
    # u,v,T = u.load(),v.load(),T.load()
    uhres,ulres,thres,tlres = hreslres(u,v,T,coarse_grain_u,coarse_grain_t,**kwargs)

    ugridforcings = {f"S{key}": _subgrid_forcing(uhres,ulres,key,coarse_grain_u,) for key in "u v".split()}
    ugridfields = {key:val for key,val in ulres.items() if "dlon" not in key and "dlat" not in key}

    ugridoutputs = concat(**dict(ugridfields,**ugridforcings))

    tgridforcings = {f"S{key}": _subgrid_forcing(thres,tlres,key,coarse_grain_t,) for key in "T"}
    tgridfields = {key:val for key,val in tlres.items() if key == "T" }


    tgridoutputs = concat(**dict(tgridfields,**tgridforcings))

    ugridoutputs = ugridoutputs.rename({'lat':'ulat','lon':'ulon'})
    tgridoutputs = tgridoutputs.rename({'lat':'tlat','lon':'tlon'})

    outputs = xr.merge([ugridoutputs,tgridoutputs])
    return outputs
