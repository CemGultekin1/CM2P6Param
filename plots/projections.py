import matplotlib.pyplot as plt

import xarray as xr
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def cartopy_plot(data,lons,lats,titles,nrow,ncol,figsize,suptitle,filename,kwargs):
    fig=plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    for i,(u,lon,lat,title) in enumerate(zip(data,lons,lats,titles)):
        cartopy_subplot(fig,nrow,ncol,i+1,u,lon,lat,title,**kwargs[i])
    fig.suptitle(suptitle)
    fig.savefig(filename,)

def imshow_plot(data,lons,lats,titles,nrow,ncol,figsize,suptitle,filename,kwargs):
    # print(suptitle)
    fig,axs=plt.subplots(nrow,ncol,figsize=figsize)
    def pick_ax(i):
        if nrow==1 and ncol==1:
            return axs
        elif nrow==1 or ncol==1:
            return axs[i]
        else:
            ii= i%ncol
            jj= i//ncol
            return axs[jj,ii]
    fig.patch.set_facecolor('white')
    for i,(u,lon,lat,title) in enumerate(zip(data,lons,lats,titles)):
        imshow_subplot(fig,pick_ax(i),u,lon,lat,title,**kwargs[i])
    fig.suptitle(suptitle)
    fig.savefig(filename,)
    plt.close(fig)


def line_plots(data,titles,nrow,ncol,figsize,suptitle,filename,kwargs):
    # print(suptitle)
    fig,axs=plt.subplots(nrow,ncol,figsize=figsize)
    def pick_ax(i):
        if nrow==1 and ncol==1:
            return axs
        elif nrow==1 or ncol==1:
            return axs[i]
        else:
            ii= i%ncol
            jj= i//ncol
            return axs[jj,ii]
    fig.patch.set_facecolor('white')
    for i,(u,title) in enumerate(zip(data,titles)):
        line_subplot(pick_ax(i),u,title,**kwargs[i])
    # fig.suptitle(suptitle)
    fig.savefig(filename,)
    plt.close(fig)

def line_subplot(ax,us,title,**kwargs):
    for i,u in enumerate(us):
        kwargs_ = {}
        if "label" in kwargs:
            kwargs_["label"] = kwargs["label"][i]
        if "color" in kwargs:
            kwargs_["color"] = kwargs["color"][i]
        ax.plot(u,**kwargs_)
    ax.set_title(title)
    ax.legend()



def imshow_subplot(fig,ax,u,lon,lat,title,**kwargs):
    if "vmin" in kwargs:
        vmin = kwargs["vmin"]
        vmax = kwargs["vmax"]
    else:
        vmax = np.amax(np.abs(u[u==u]))
        vmin = np.amin(np.abs(u[u==u]))
        if vmin<=0 and vmax >=0:
                ext = np.maximum(np.abs(vmin),np.abs(vmax))
                vmin = -ext
                vmax = ext
    kwargs_ = {"vmin":vmin,"vmax":vmax}
    if "extent" in kwargs:
        kwargs_["extent"] = kwargs["extent"]
    # print(kwargs_)
    neg=ax.imshow(u,cmap='coolwarm',**kwargs_)#extent = [lon[0],lon[-1],lat[0],lat[-1]])#cmap=color,extent=[xmin,xmax,ymin,ymax],vmin=MINS[seli],vmax=MAXS[seli])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar=fig.colorbar(neg,cax=cax)
    if 'units'in kwargs:
        cbar.ax.set_ylabel(kwargs['units'])
    if 'yticks' in kwargs:
        cbar.ax.set_yticklabels(kwargs['yticks'])
    ax.set_title(title)
def cartopy_subplot(fig,nrow,ncol,index,u,lons,lats,title,\
    exts=[],lognorm= False,cmap='coolwarm',unit = 'x$10^{-7}$ m/$s^2$',colorbar = True):
    ax = plt.subplot(nrow,ncol,index, projection=ccrs.Robinson())
    if len(exts)==0:
        umax=u[u==u].max()
        umin=u[u==u].min()
    else:
        umin,umax=exts[0],exts[1]
    ds=xr.Dataset(data_vars=dict(u=(["latitude","longitude"],u)),\
                            coords=dict(latitude=lats,longitude=lons))
    if lognorm:
        norm = mcolors.LogNorm(vmin=umin,vmax=umax)
    else:
        norm = mcolors.Normalize(vmin=umin,vmax=umax)
    cax=ds.u.plot(ax=ax, transform = ccrs.PlateCarree(),\
                  cmap=cmap,\
                  norm=norm,\
                  add_colorbar=False)
    ax.set_title(title)
    ax.coastlines()

    # tickvals=[10**(-(i-1)) for i in range(8)]
    # tickstr=[str(tt) for tt in tickvals]
    if colorbar:
        cbar = fig.colorbar(cax, fraction=0.035, pad=0.04)#ticks=tickvals,
        # cbar.ax.set_yticklabels(tickstr)
        cbar.ax.set_ylabel(unit)
