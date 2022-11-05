import copy
from typing import Dict, List, Tuple
import torch
from data.low_res import CM2p6Dataset,DividedDomain
from data.geography import frequency_encoded_latitude
import numpy as np
from data.vars import get_var_mask_name
import xarray as xr
from utils.xarray import tonumpydict
def determine_ndoms(*args,**kwargs):
    arglens = [1]
    for i in range(len(args)):
        if isinstance(args[i],list):
            arglens.append(len(args[i]))
    for key,val in kwargs.items():
        if isinstance(kwargs[key],list):
            arglens.append(len(kwargs[key]))
    return  int(np.amax(arglens))
class MultiDomain(CM2p6Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ndoms = determine_ndoms(*args,**kwargs)
        self.domain_datasets  : List[DividedDomain]= []
        self.shapes = []
        self.parts = []
        def read_value(var,i):
            if isinstance(var,list):
                return var[i]
            else:
                return var
        for i in range(self.ndoms):
            args_ = [read_value(arg,i) for arg in args]
            kwargs_ = {key:read_value(val,i) for key,val in kwargs.items()}
            self.domain_datasets.append(DividedDomain(*args_,**kwargs_))
            self.shapes.append(self.domain_datasets[-1].shapes)
            self.parts.append(self.domain_datasets[-1].parts)
        self.linsupres = self.domain_datasets[0].linsupres
        self.half_spread = self.domain_datasets[0].half_spread
        self.normalization = kwargs.pop("normalization","standard")
        self.var_grouping = kwargs.pop('var_grouping')


    def set_time_constraint(self,t0,t1):
        super().set_time_constraint(t0,t1)
        for i in range(self.ndoms):
            self.domain_datasets[i].set_time_constraint(t0,t1)

    def time_slice(self,t0,t1)->'MultiDomain':
        x = copy.deepcopy(self)
        x.set_time_constraint(t0,t1)
        return x

    def set_half_spread(self,num):
        self.half_spread = num
        for dds in self.domain_datasets:
            dds.set_half_spread(num)

    def get_max_shape(self,):
        lat,lon = 0,0
        for i in range(self.ndoms):
            lat_,lon_ = self.domain_datasets[i].get_max_shape()
            lat = np.maximum(lat_,lat)
            lon = np.maximum(lon_,lon)
        return lat,lon

    def time2index(self,t):
        a = 0
        for nlon,nlat in self.parts:
            a+=nlon*nlat
        return a*t,a*(t+1)
    def get_lat_features(self,lats):
        posdict = self.locate(lats[0],lats[-1],lat = True)
        (n0,_),n = posdict['locs'],posdict["len"]
        slc = slice(n0,len(lats)+n0)
        abslat,signlat = frequency_encoded_latitude(n,self.half_spread*2+1)
        return np.cos(abslat[slc]),np.cos(signlat[slc])
    def append_lat_features(self,outs):
        key = list(outs.keys())[0]
        lats = outs[key].u.lat.values
        abslat,signlat = self.get_lat_features(lats)
        n = len(outs[key].u.lon)
        abslat = abslat.reshape(-1,1)@np.ones((1,n))
        signlat = signlat.reshape(-1,1)@np.ones((1,n))
        latfeats = xr.Dataset(
            data_vars = dict(
                abslat = (["lat","lon"],abslat),
                signlat = (["lat","lon"],signlat),
            ),
            coords = dict(
                lon = outs[key].u.lon,
                lat = outs[key].u.lat
            )
        )
        outs['lat_feats'] = latfeats
        return outs
    def __len__(self,):
        return sum([len(dom) for dom in self.domain_datasets])
    def __getitem__(self,i):
        idom  = i%self.ndoms
        ds = self.domain_datasets[idom][i//self.ndoms]
        ds = ds.assign_coords(**{'domain_id':np.array(idom)})
        return ds


class MultiDomainDataset(MultiDomain):
    def __init__(self,*args,scalars_dict = None,latitude = False,temperature = False,torch_flag = False,**kwargs):
        self.scalars_dict = scalars_dict
        self.latitude = latitude
        self.temperature = temperature
        self.running_mean = {}
        self.torch_flag = torch_flag
        super().__init__(*args,**kwargs)

    @property
    def max_shape(self,):
        return np.array(self.get_max_shape())

    @property
    def sslice(self,):
        return slice(self.half_spread,-self.half_spread)

    def pad(self,data_vars:dict,coords:dict):
        for name in data_vars.keys():
            dims,vals = data_vars[name]
            if 'lat' not in dims or 'lon' not in dims:
                continue
            
            pad = self.max_shape - np.array(vals.shape)
            if name in self.forcing_names and self.half_spread>0:
                vals =  vals[self.sslice,self.sslice]
            vals = np.pad(vals,pad_width = ((0,pad[0]),(0,pad[1])),constant_values = np.nan)
            data_vars[name] = (dims,vals)
        
        def pad_coords(coords,slice_flag = False):
            lat = coords['lat']
            pad = self.max_shape[0] - len(lat)
            coords['lat_pad'] = pad
            lat = np.pad(lat,pad_width = ((0,pad),),constant_values = 0)
            if slice_flag:
                lat = lat[self.sslice]
            coords['lat'] = lat

            lon = coords['lon']
            pad = self.max_shape[1] - len(lon)
            coords['lon_pad'] = pad
            lon = np.pad(lon,pad_width = ((0,pad),),constant_values = 0)
            if slice_flag:
                lon = lon[self.sslice]
            coords['lon'] = lon
            return coords
        
        forcing_coords = pad_coords(copy.deepcopy(coords),slice_flag=True)
        coords = pad_coords(coords,slice_flag=False)
        
        return data_vars,coords,forcing_coords

    def add_lat_features(self,data_vars,coords):
        lats = coords['lat']
        lons = coords['lon']
        abslat,signlat = self.get_lat_features(lats)
        data_vars['abs_lat'] = (['lat','lon'], abslat.reshape([-1,1]) @ np.ones((1,len(lons))))
        data_vars['sign_lat'] = (['lat','lon'],signlat.reshape([-1,1]) @ np.ones((1,len(lons))))
        return data_vars
    def group_variables(self,data_vars):
        groups = []
        # normalization_groups = []
        for vargroup in self.var_grouping:
            valdict = {}
            # valnormalization = {}
            for varname in vargroup:
                valdict[varname] = data_vars[varname]
                nvarname = f"{varname}_normalization"
                if nvarname in data_vars:
                    valdict[nvarname] = data_vars[nvarname]
            groups.append(valdict)
            # normalization_groups.append(valnormalization)
        # if not self.torch_flag:
            # groups.extend(normalization_groups)
        return tuple(groups)

    def group_np_stack(self,vargroups):
        return tuple([self._np_stack(vars) for vars in vargroups])
    def _np_stack(self,vals:Dict[str,Tuple[List[str],np.ndarray]]):
        v = []
        for _,val in vals.values():
            v.append(val)
        if len(v) == 0:
            return np.empty(0)
        else:
            return np.stack(v,axis =0)
    def group_to_torch(self,vargroups):
        return tuple([self._to_torch(vars) for vars in vargroups])
    def _to_torch(self,vals:np.array,type = torch.float32):
        return torch.from_numpy(vals).type(type)
    def normalize(self,data_vars,coords):
        keys_list = tuple(data_vars.keys())
        for key in keys_list:
            a,b = 0,1
            if self.scalars_dict is not None:
                if key in self.scalars_dict:
                    a,b = self.scalars_dict[key]
            if not self.torch_flag:
                coords['normalization'] = ['mean','std']
                data_vars[f"{key}_normalization"] =(['normalization'],np.array([a,b]))
            dims,vals = data_vars[key]
            vals = (vals - a)/b
            data_vars[key] = (dims,vals)

        return data_vars,coords

    def mask(self,data_vars):
        keys_list = tuple(data_vars.keys())
        for key in keys_list:
            dims,f = data_vars[key]
            if not ('lat' in dims and 'lon' in dims):
                continue
            mask = f==f
            f[~mask] = 0
            varmask = get_var_mask_name(key)
            data_vars[varmask] = (dims,mask)
            if not self.torch_flag:
                data_vars[f"{varmask}_normalization"] = (['normalization'],np.array([0,1]))
        return data_vars
    def fillna(self,values):
        for key,v in values.items():
            v[v!=v] = 0
            values[key] = v
        return values
    def __getitem__(self,i):
        outs = super().__getitem__(i)
        data_vars,coords = tonumpydict(outs)

        if self.latitude:
            data_vars = self.add_lat_features(data_vars,coords)

        data_vars,coords = self.normalize(data_vars,coords)
        data_vars,coords,forcing_coords = self.pad(data_vars,coords)
        data_vars = self.mask(data_vars)
        grouped_vars = self.group_variables(data_vars)

        if self.torch_flag:
            grouped_vars = self.group_np_stack(grouped_vars)
            return self.group_to_torch(grouped_vars)
        else:
            grouped_vars = list(grouped_vars)
            grouped_vars.append(coords)
            grouped_vars.append(forcing_coords)
            return tuple(grouped_vars)
