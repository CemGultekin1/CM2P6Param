import copy
from typing import Dict, List
import torch
from data.gcm_forcing import CM2p6Dataset
from data.geography import frequency_encoded_latitude
import numpy as np
from data.vars import get_var_mask_name
import xarray as xr
from data.gcm_lsrp import DividedDomain

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


    def outs2numpydict(self,outs):
        vald = {}
        for key,val in outs.items():
            for varname,vals in val.data_vars.items():
                name = f"{key}_{varname}"
                vald[name] = (vals.values,vals.lat.values,vals.lon.values)
        return vald

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
        return dict(self.domain_datasets[idom][i//self.ndoms], **dict(idom= idom))


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

    def outs2numpydict_latlon(self,outs,):
        vald = {}
        for key,val in outs.items():
            for varname,vals in val.data_vars.items():
                # name = f"{key}_{varname}"
                vald[varname] = dict( val = vals.values,lat = vals.lat.values.reshape([-1]) ,lon = vals.lon.values.reshape([-1]) )
        return vald

    def shapeshift(self,values):
        for name in values.keys():
            if not isinstance(values[name],dict):
                continue
            pad = self.max_shape - np.array(values[name]['val'].shape)
            if name in self.forcing_names and self.half_spread>0:
                values[name]['val'] = values[name]['val'][self.sslice,self.sslice]
                values[name]['lon'] = values[name]['lon'][self.sslice]
                values[name]['lat'] = values[name]['lat'][self.sslice]
            values[name]['val'] = np.pad(values[name]['val'],pad_width = ((0,pad[0]),(0,pad[1])),constant_values = np.nan)
            values[name]['lon'] = np.pad(values[name]['lon'],pad_width = ((0,pad[1]),),constant_values = np.nan)
            values[name]['lat'] = np.pad(values[name]['lat'],pad_width = ((0,pad[0]),),constant_values = np.nan)

        return values

    def add_lat_features(self,vals,):
        key0 = self.var_grouping[0][0]
        lats = vals[key0]['lat']
        lons = vals[key0]['lon']
        abslat,signlat = self.get_lat_features(lats)
        vals['abs_lat'] = dict(val = abslat.reshape([-1,1]) @ np.ones((1,len(lons))),lat = lats,lon = lons)
        vals['sign_lat'] = dict(val = signlat.reshape([-1,1]) @ np.ones((1,len(lons))),lat = lats,lon = lons)
        return vals
    def group_variables(self,values):
        groups = []
        for vargroup in self.var_grouping:
            valdict = {}
            for varname in vargroup:
                valdict[varname] = values[varname]
            groups.append(valdict)
        return tuple(groups)

    def group_np_stack(self,vargroups):
        return tuple([self._np_stack(vars) for vars in vargroups])
    def _np_stack(self,vals:Dict[str,Dict[str,np.array]]):
        v = []
        for val in vals.values():
            if isinstance(val,dict):
                v.append(val['val'])
            else:
                v.append(val)
        return np.stack(v,axis =0)
    def group_to_torch(self,vargroups):
        return tuple([self._to_torch(vars) for vars in vargroups])
    def _to_torch(self,vals:np.array,type = torch.float32):
        return torch.from_numpy(vals).type(type)
    def normalize(self,values):
        keys_list = tuple(values.keys())
        for key in keys_list:
            a,b = 0,1
            if self.scalars_dict is not None:
                if key in self.scalars_dict:
                    a,b = self.scalars_dict[key]
            if isinstance(values[key],dict):
                if not self.torch_flag:
                    values[key]['normalization'] = np.array([a,b])
                values[key]['val'] = (values[key]['val'] - a)/b

        return values

    def mask(self,values):
        keys_list = tuple(values.keys())
        for key in keys_list:
            if not isinstance(values[key],dict):
                continue
            f = values[key]['val']
            mask = f==f
            values[key]['val'][~mask] = 0
            varmask = get_var_mask_name(key)
            values[varmask] = {key_:val for key_,val in values[key].items()}
            values[varmask]['val'] = mask
            values[varmask]['normalization'] = np.array([0,1])
        return values
    def fillna(self,values):
        for key,v in values.items():
            v[v!=v] = 0
            values[key] = v
        return values
    # def remove_temperature(self,values:Dict[str,np.array]):
    #     rmkeys = []
    #     for key in values:
    #         if 'T' in key:
    #             rmkeys.append(key)
    #     for key in rmkeys:
    #         values.pop(key)
    #     return values
    def __getitem__(self,i):
        outs = super().__getitem__(i)
        location = {key: outs.pop(key) for key in ['ilat','ilon','itime','idom','depth']}
        values = self.outs2numpydict_latlon(outs)
        values = dict(values,**location)

        if self.latitude:
            values = self.add_lat_features(values)

        values = self.normalize(values)
        values = self.shapeshift(values)
        values = self.mask(values)
        grouped_vars = self.group_variables(values)

        if self.torch_flag:
            grouped_vars = self.group_np_stack(grouped_vars)
            return self.group_to_torch(grouped_vars)
        else:
            for i in range(len(grouped_vars)):
                for key in grouped_vars[i]:
                    if isinstance(grouped_vars[i][key],dict):
                        for key_ in  grouped_vars[i][key]:
                            grouped_vars[i][key][key_] = torch.tensor(grouped_vars[i][key][key_],dtype = torch.float32)
                    else:
                        grouped_vars[i][key] = torch.tensor(grouped_vars[i][key],dtype = torch.float32)
            return grouped_vars
