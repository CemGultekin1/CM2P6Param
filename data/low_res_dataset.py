import copy
from typing import Dict, List, Tuple
import torch
from data.low_res import SingleDomain
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
    for key,_ in kwargs.items():
        if isinstance(kwargs[key],list):
            arglens.append(len(kwargs[key]))
    return  int(np.amax(arglens))
class MultiDomain(SingleDomain):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.var_grouping = kwargs.pop('var_grouping')
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
    


class MultiDomainDataset(MultiDomain):
    def __init__(self,*args,scalars = None,latitude = False,temperature = False,torch_flag = False,**kwargs):
        self.scalars = scalars
        self.latitude = latitude
        self.temperature = temperature
        self.torch_flag = torch_flag
        super().__init__(*args,**kwargs)


    @property
    def sslice(self,):
        return slice(self.half_spread,-self.half_spread)

    def pad(self,data_vars:dict,coords:dict):
        for name in data_vars.keys():
            dims,vals = data_vars[name]
            if 'lat' not in dims or 'lon' not in dims:
                continue
            pad = (0,0)
            if name in self.forcing_names and self.half_spread>0:
                vrshp = list(vals.shape)
                vals = vals.reshape([-1]+ vrshp[-2:])
                vals =  vals[:,self.sslice,self.sslice]
                vals = vals.reshape(vrshp[:-2] + list(vals.shape[-2:]))
            padtuple = (len(vals.shape)-2)*[(0,0)] + [(0,pad[0]),(0,pad[1])]
            vals = np.pad(vals,pad_width = tuple(padtuple),constant_values = np.nan)
            data_vars[name] = (dims,vals)
        
        def pad_coords(coords,slice_flag = False):
            lat = coords['lat']
            pad = 0
            coords['lat_pad'] = pad
            lat = np.pad(lat,pad_width = ((0,pad),),constant_values = 0)
            if slice_flag:
                lat = lat[self.sslice]
            coords['lat'] = lat

            lon = coords['lon']
            pad = 0
            coords['lon_pad'] = pad
            lon = np.pad(lon,pad_width = ((0,pad),),constant_values = 0)
            if slice_flag:
                lon = lon[self.sslice]
            coords['lon'] = lon
            return coords
        
        forcing_coords = pad_coords(copy.deepcopy(coords),slice_flag=self.half_spread>0)
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
                for suff in '_mean _std'.split():
                    nvarname = varname + suff
                    if nvarname in data_vars:
                        valdict[nvarname] = data_vars[nvarname]
            groups.append(valdict)
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
            dims,vals = data_vars[key]
            if 'lat' not in dims or 'lon' not in dims:
                continue
            shp = {d:len(coords[d]) for d in dims}
            newdims = {key:None for key in shp}
            if 'lon' in shp:
                shp['lon'] = 1
                newdims.pop('lon')
            if 'lat' in shp:
                shp['lat'] = 1
                newdims.pop('lat')
            shp0 = [shp[key] for key in newdims]
            shp1 = list(shp.values())
            newdims = list(newdims.keys())
            
            
            a,b = np.zeros(shp0),np.ones(shp0)
            if self.scalars is not None:
                if f"{key}_mean" in self.scalars:
                    a = self.scalars[f"{key}_mean"].values
                    b = self.scalars[f"{key}_std"].values
                    a = a.reshape(shp0)
                    b = b.reshape(shp0)

            

            if not self.torch_flag:
                data_vars[f"{key}_mean"] = (newdims,a)
                data_vars[f"{key}_std"] = (newdims,b)
            vals = (vals - a.reshape(shp1))/b.reshape(shp1)
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
