#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:15:35 2020
@author: arthur
"""

from transforms.subgrid_forcing import gcm_lsrp_subgrid_forcing, gcm_subgrid_forcing, scipy_subgrid_forcing

import xarray as xr
from scipy.ndimage import gaussian_filter
import numpy as np
import gcm_filters as gcm

def advections(u_v_field: xr.Dataset, grid_data: xr.Dataset,**kwargs):
    """
    Return the advection terms corresponding to the passed velocity field.
    Note that the velocities sit on U-grids
    Parameters
    ----------
    u_v_field : xarray dataset
        Velocity field, must contains variables usurf and vsurf.
    grid_data : xarray dataset
        Dataset with grid details, must contain variables dxu and dyu.
    Returns
    -------
    advections : xarray dataset
        Advection components, under variable names adv_x and adv_y.
    """
    
    if kwargs.get('periodic_diff',True):
        gradient_x = u_v_field - u_v_field.roll(xu_ocean = 1)
        gradient_y = u_v_field - u_v_field.roll(yu_ocean = 1)
    else:
        gradient_x = u_v_field.diff(dim='xu_ocean') 
        gradient_y = u_v_field.diff(dim='yu_ocean')

    
    
    dxu = grid_data['dxu'].copy()
    dyu = grid_data['dyu'].copy()
    gradient_x = gradient_x / dxu
    gradient_y = gradient_y/dyu

    # Interpolate back the gradients
    # interp_coords = dict(xu_ocean=u_v_field.coords['xu_ocean'],
    #                      yu_ocean=u_v_field.coords['yu_ocean'])
    # gradient_x = gradient_x.interp(interp_coords)
    # gradient_y = gradient_y.interp(interp_coords)
    u, v = u_v_field['usurf'], u_v_field['vsurf']
    adv_x = u * gradient_x['usurf'] + v * gradient_y['usurf']
    adv_y = u * gradient_x['vsurf'] + v * gradient_y['vsurf']
    result = xr.Dataset({'adv_x': adv_x, 'adv_y': adv_y})
    # TODO check if we can simply prevent the previous operation from adding
    # chunks
    #result = result.chunk(dict(xu_ocean=-1, yu_ocean=-1))
    return result


def spatial_filter(data: np.ndarray, filter_fun):
    """
    Apply a gaussian filter along all dimensions except first one, which
    corresponds to time.
    Parameters
    ----------
    data : numpy array
        Data to filter.
    sigma : float
        Unitless scale of the filter.
    Returns
    -------
    result : numpy array
        Filtered data.
    """

    result = np.zeros_like(data)
    for t in range(data.shape[0]):
        data_t = data[t, ...]
        result_t = filter_fun(data_t)
        result[t, ...] = result_t
    return result


def spatial_filter_dataset(dataset: xr.Dataset, grid_info: xr.Dataset,
                           filter_scale: float,**kwargs):
    """
    Apply spatial filtering to the dataset across the spatial dimensions.
    Parameters
    ----------
    dataset : xarray dataset
        Dataset to which filtering is applied. Time must be the first
        dimension, whereas spatial dimensions must come after.
    grid_info : xarray dataset
        Dataset containing details on the grid, in particular must have
        variables dxu and dyu.
    sigma : float
        Scale of the filtering, same unit as those of the grid (often, meters)
    Returns
    -------
    filt_dataset : xarray dataset
        Filtered dataset.
    """
    area_u = grid_info['dxu'] * grid_info['dyu'] / 1e8
    if kwargs.get('gcm_filtering',False):
        # wet_mask = xr.where(np.isnan(dataset.usurf),1,0)
        specs = {
            'filter_scale': filter_scale*np.sqrt(12),
            'dx_min': 1,
            'filter_shape':gcm.FilterShape.GAUSSIAN,
            'grid_type':gcm.GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED,
            'grid_vars':dict(area = area_u,wet_mask = grid_info['wet_mask']),
        }
        gf = gcm.Filter(**specs)

        specs = {
            'filter_scale': filter_scale*np.sqrt(12),
            'dx_min': 1,
            'filter_shape':gcm.FilterShape.GAUSSIAN,
            'grid_type':gcm.GridType.REGULAR,
        }
        gf0 = gcm.Filter(**specs)
        norm = gf0.apply(area_u,dims = ['yu_ocean','xu_ocean'])
        filtered = gf.apply(dataset,dims = ['yu_ocean','xu_ocean'])*area_u
    else:
        filter_fun = lambda x : gaussian_filter(x, filter_scale, mode='wrap')
        # Normalisation term, so that if the quantity we filter is constant
        # over the domain, the filtered quantity is constant with the same value
        dataset = dataset* area_u
        norm = xr.apply_ufunc(lambda x: filter_fun(x),
                            area_u, dask='parallelized', output_dtypes=[float, ])
        filtered = xr.apply_ufunc(lambda x: filter_fun(x), dataset,
                                dask='parallelized', output_dtypes=[float, ])
    return filtered / norm


def eddy_forcing(u_v_dataset : xr.Dataset, grid_data: xr.Dataset,**kwargs) -> xr.Dataset:
    """
    Compute the sub-grid forcing terms.
    Parameters
    ----------
    u_v_dataset : xarray dataset
        High-resolution velocity field.
    grid_data : xarray dataset
        High-resolution grid details.
    scale : float
        Scale, in meters, or factor, if scale_mode is set to 'factor'
    method : str, optional
        Coarse-graining method. The default is 'mean'.
    nan_or_zero: str, optional
        String set to either 'nan' or 'zero'. Determines whether we keep the
        nan values in the initial surface velocities array or whether we
        replace them by zeros before applying the procedure.
        In the second case, remaining zeros after applying the procedure will
        be replaced by nans for consistency.
        The default is 'zero'.
    scale_mode: str, optional
        DEPRECIATED, should always be left as 'factor'
    Returns
    -------
    forcing : xarray dataset
        Dataset containing the low-resolution velocity field and forcing.
    """
    scale = kwargs.get('scale',4)
    scale_filter = scale / 2
    # High res advection terms
    adv = advections(u_v_dataset, grid_data,**kwargs)
    # Filtered advections
    filtered_adv = spatial_filter_dataset(adv, grid_data, scale_filter,**kwargs)
    # Filtered u,v field and temperature
    u_v_filtered = spatial_filter_dataset(u_v_dataset, grid_data, scale_filter,**kwargs)
    # Advection term from filtered velocity field
    adv_filtered = advections(u_v_filtered, grid_data,**kwargs)
    # Forcing
    forcing = adv_filtered - filtered_adv
    forcing = forcing.rename({'adv_x': 'S_x', 'adv_y': 'S_y'})
    # Merge filtered u,v, temperature and forcing terms
    forcing = forcing.merge(u_v_filtered)
    # Coarsen
    print('scale factor: ', scale)
    
    if not kwargs.get('greedy_coarse_grain',False):
        forcing_coarse = forcing.coarsen({'xu_ocean': int(scale),
                                        'yu_ocean': int(scale)},
                                        boundary='trim').mean()

    else:
        forcing_coarse = forcing.fillna(0).coarsen({'xu_ocean': int(scale),
                                        'yu_ocean': int(scale)},
                                        boundary='trim').mean()

        cwet_mask = grid_data.wet_mask.coarsen({'xu_ocean': int(scale),
                                        'yu_ocean': int(scale)},
                                        boundary='trim').mean()
        forcing_coarse = forcing_coarse/cwet_mask
    return forcing_coarse
    
def get_coarse_grid_data(grid_data,scale):
    coarse_grid_data = None
    for typ in 'x y'.split():
        cgsep = grid_data[f'd{typ}u'].coarsen(xu_ocean = scale,yu_ocean = scale).sum()/scale
        cgsep.name = f'cd{typ}u'
        if coarse_grid_data is None:
            coarse_grid_data = cgsep
        else:
            coarse_grid_data = xr.merge([coarse_grid_data,cgsep])
    # plot_ds(coarse_grid_data,'coarse_grid_data',dims = ['xu_ocean','yu_ocean'])
    # raise Exception
    return coarse_grid_data
def main():
    import os
    scale = 4
    root = '/scratch/zanna/data/cm2.6/'
    file = 'surface.zarr'
    path = os.path.join(root,file)
    sl = dict(xu_ocean = slice(800,1800),yu_ocean = slice(800,1800))
    u_v_dataset = xr.open_zarr(path).isel(time = 0).isel(**sl)
    u_v_dataset = u_v_dataset.drop('surface_temp')
    path = os.path.join(root,'GFDL_CM2_6_grid.nc')
    grid_data = xr.open_dataset(path).isel(**sl)
    grid_data['wet_mask'] = xr.where(np.isnan(u_v_dataset.usurf),0,1)
    grid_data['area'] = grid_data.dxu*grid_data.dyu

    gsbf = gcm_subgrid_forcing(scale,grid_data,dims = ['yu_ocean','xu_ocean'],grid_separation = 'dyu dxu'.split(),momentum = 'usurf vsurf'.split())
    ssbf = scipy_subgrid_forcing(scale,grid_data,dims = ['yu_ocean','xu_ocean'],grid_separation = 'dyu dxu'.split(),momentum = 'usurf vsurf'.split())

    org_forcing = eddy_forcing(u_v_dataset, grid_data,scale = scale)

    forcings_list = []
    forcings_list.append(gsbf(u_v_dataset,'usurf vsurf'.split(),'S_x S_y'.split()))
    forcings_list.append(ssbf(u_v_dataset,'usurf vsurf'.split(),'S_x S_y'.split()))

    cmpr_forcings = org_forcing
    for type_num,forcings in enumerate(forcings_list):
        names = list(forcings.data_vars.keys())
        # dforcings = dforcings.rename({n:f"log10(|{n}_{type_num+1} - {n}|)" for n in names})
        forcings = forcings.rename({n:f"{n}_{type_num+1}" for n in names})
        cmpr_forcings = xr.merge([cmpr_forcings,forcings])
        # print(names)
            

    
    from utils.xarray import plot_ds,drop_unused_coords
    def plot_forcing(forcing,root):
        forcing = drop_unused_coords(forcing)
        nms = list(forcing.data_vars.keys())
        nms = np.unique([n.replace('_1','').replace('_res','') for n in nms])
        for nm in nms:
            nm1 = f"{nm}_1"
            nm2 = f"{nm}_res_1"
            u = forcing[nm]
            u1 = forcing[nm1]
            if nm2 in forcing.data_vars:
                ulsrp = forcing[nm2] + forcing[nm1]
                fs = {nm:u,f"{nm}-gcm":u1,f"{nm}-lsrp":ulsrp}
                plot_ds(fs,root + nm,ncols = 3,dims = ['xu','yu'])
            else:
                fs = {nm:u,f"{nm}-gcm":u1}
                plot_ds(fs,root + nm,ncols = 2,dims = ['xu','yu'])
    def plot_forcing_(forcing,root):
        forcing = drop_unused_coords(forcing)
        nms = list(forcing.data_vars.keys())
        n = len(forcings_list)
        for i in range(n):
            nms = [n.replace(f'_{i+1}','') for n in nms]
        nms = np.unique(nms)
        for nm in nms:
            fs = {}
            fs[nm] = forcing[nm]
            fs = dict(fs,**{f"{nm}_{i+1}":\
                forcing[f"{nm}_{i+1}"] for i in range(n)})
                # np.log10(np.abs(forcing[nm] - forcing[f"{nm}_{i+1}"])) - \
                # np.log10(np.mean(np.abs(forcing[nm].fillna(0))))
                    # for i in range(n)})
                
            
            plot_ds(fs,root + nm,ncols = n+1,dims = ['xu','yu'])
    plot_forcing_(cmpr_forcings,'saves/plots/filtering/local4_')

if __name__ == '__main__':
    main()