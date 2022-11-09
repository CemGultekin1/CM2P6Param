import itertools
from typing import Callable
from data.coords import TIMES
from transforms.coarse_grain_inversion import coarse_grain_inversion_weights, coarse_grain_projection
from utils.paths import average_highres_fields_path, average_lowhres_fields_path, coarse_graining_projection_weights_path
from utils.slurm import flushed_print
# from utils.slurm import flushed_print
from utils.xarray import tonumpydict
import xarray as xr
from transforms.coarse_grain import coarse_graining_2d_generator, hreslres
from transforms.grids import get_grid_vars, logitudinal_expansion, logitudinal_expansion_dataset, trim_expanded_longitude, ugrid2tgrid
from transforms.subgrid_forcing import subgrid_forcing
import numpy as np

class HighResCm2p6:
    ds : xr.Dataset
    sigma : int
    half_spread : int
    coarse_grain : Callable
    initiated : bool
    def __init__(self,ds:xr.Dataset,sigma,*args,**kwargs):
        self.ds = ds.copy()
        self.sigma = sigma
        self.coarse_grain = None
        self.initiated = False
    @property
    def depth(self,):
        return self.ds.depth

    def is_deep(self,):
        return self.depth[0] > 1e-3
    def __len__(self,):
        return len(self.ds.time)
    @property
    def coarse_graining_half_spread(self,):
        return int(self.sigma*6)
    
    @property
    def coarse_graining_crop(self,):
        return 5
    def time_depth_indices(self,i):
        di = i%len(self.ds.depth)
        ti = i//len(self.ds.depth)
        return ti,di
    def get_hres(self,i,fillna = False):
        _,di,ti = self.time_depth_indices(i)
        ds = self.ds.isel(time = ti,depth = di)
        # ds = ds.isel(ulon = slice(1000,1300),ulat = slice(1000,1300),tlon = slice(1000,1300),tlat = slice(1000,1300))
        
        if fillna:
            ds = ds.fillna(0)
        u,v,T = ds.u,ds.v,ds.T
        ugrid,tgrid = get_grid_vars(ds)
        u = u.rename(ulat = "lat",ulon = "lon")
        v = v.rename(ulat = "lat",ulon = "lon")
        T = T.rename(tlat = "lat",tlon = "lon")
        return ds.time.values,ds.depth.values,u.load(),v.load(),T.load(),ugrid,tgrid
    def get_gridvars(self,):
        return None,None
    def init_coarse_graining(self,i):
        time,depth,u,v,T,ugrid,tgrid=self.get_hres(i)

        u = logitudinal_expansion(u,self.coarse_graining_half_spread)
        v = logitudinal_expansion(v,self.coarse_graining_half_spread)
        T = logitudinal_expansion(T,self.coarse_graining_half_spread)

        ugrid = logitudinal_expansion_dataset(ugrid,self.coarse_graining_half_spread)
        tgrid = logitudinal_expansion_dataset(tgrid,self.coarse_graining_half_spread)

        cgu = coarse_graining_2d_generator(ugrid,self.sigma,wetmask = True)
        cgt = coarse_graining_2d_generator(tgrid,self.sigma,wetmask = True)


        dry_cgu = coarse_graining_2d_generator(ugrid,self.sigma,wetmask = False)
        dry_cgt = coarse_graining_2d_generator(tgrid,self.sigma,wetmask = False)

        self.coarse_grain =  (cgu,cgt)
        self.dry_coarse_grain =  (dry_cgu,dry_cgt)
        return time,depth,u,v,T,ugrid,tgrid
    def subgrid_forcing(self,u,v,T):
        sfds = subgrid_forcing(u,v,T,*self.coarse_grain)
        sfds = trim_expanded_longitude(sfds,expansion = self.coarse_graining_crop)
        return sfds
    def hres2lres(self,i):
        if not self.initiated:
            time,depth,u,v,T = self.init_coarse_graining(i)            
        else:
            time,depth,u,v,T = self.get_hres(i,)
            u = logitudinal_expansion(u,self.coarse_graining_half_spread)
            v = logitudinal_expansion(v,self.coarse_graining_half_spread)
            T = logitudinal_expansion(T,self.coarse_graining_half_spread)
        

        sfds = subgrid_forcing(u,v,T,*self.coarse_grain)
        sfds = trim_expanded_longitude(sfds,expansion = self.coarse_graining_crop)

        sfds = sfds.expand_dims(dim = {"time": [time]},axis=0)
        sfds = sfds.expand_dims(dim = {"depth": [depth]},axis=1)
        sfds.to_netcdf('forcings.nc')
        return sfds
    def __getitem__(self,i):
        ds = self.hres2lres(i)
        return tonumpydict(ds)

class HighResProjection:
    def __init__(self,sigma,):
        self.sigma = sigma
        self._projections = None
        self._avg_fields = None
    @property
    def projections(self,):
        if self._projections is None:
            self.load_projections()
        return self._projections
    @property
    def avg_fields(self,):
        if self._avg_fields is None:
            self.load_average_fields()
        return self._avg_fields
    def load_projections(self,):
        path = coarse_graining_projection_weights_path(self.sigma)
        projections = xr.open_dataset(path)
        self._projections = projections.load()
    def load_average_fields(self,):
        path = average_lowhres_fields_path(self.sigma,False)
        _surf_avg_fields = xr.open_dataset(path)#.isel(ulon = slice(1000,1300),ulat = slice(1000,1300),tlon = slice(1000,1300),tlat = slice(1000,1300))
        for key in list(_surf_avg_fields.data_vars):
            _surf_avg_fields[key] =  _surf_avg_fields[key].expand_dims(dim = {'depth':[0]},axis=0)
        path = average_lowhres_fields_path(self.sigma,True)
        _deep_avg_fields = xr.open_dataset(path)#.isel(ulon = slice(1000,1300),ulat = slice(1000,1300),tlon = slice(1000,1300),tlat = slice(1000,1300))
        self._avg_fields = xr.merge([_surf_avg_fields,_deep_avg_fields])

    def project(self,v,**kwargs):
        return coarse_grain_projection(v,self.projections,**kwargs)


def fillnarandom(x:xr.DataArray):
    xv = x.values.copy()
    rv = np.random.randn(*xv.shape)
    rv[~np.isnan(xv)] = 0
    return xr.DataArray(
        data = rv,
        dims = x.dims,
        coords = x.coords,
        name = x.name
    )
class SubgridForcingInputs:
    def __init__(self,uvars,tvars):
        return
class ProjectedHighResCm2p6(HighResCm2p6):
    def __init__(self, ds: xr.Dataset, sigma, *args, **kwargs):
        super().__init__(ds, sigma, *args, **kwargs)
        self.projections = HighResProjection(sigma,)
        self.wetmask = None
        a,b = kwargs.get('section')
        nt = len(self.ds.time)
        time_secs = np.linspace(0,nt,b+1).astype(int)
        t0 = int(time_secs[a])
        t1 = int(time_secs[a+1])
        self.ds = self.ds.isel(time = slice(t0,t1))
        # print(self.ds.time.values[[0,-1]])
        # raise Exception
    def __len__(self,):
        return len(self.ds.depth)*len(self.ds.time)
        # return len(self.projections.avg_fields.depth)*len(self.ds.depth)*len(self.ds.time)
    def time_depth_indices(self,i):
        # pdi = i%len(self.projections.avg_fields.depth)
        # i = i//len(self.projections.avg_fields.depth)
        
        di = i%len(self.ds.depth)
        ti = i//len(self.ds.depth)

        pdi = np.argmin(np.abs(self.projections.avg_fields.depth.values - self.ds.depth.values[di]))
        return pdi,di,ti
    def hres2lres(self,i,maskpurposed:bool = False):
        if not self.initiated:
            time,depth,u,v,T,ugrid,tgrid = self.init_coarse_graining(i)
        else:
            time,depth,u,v,T,ugrid,tgrid = self.get_hres(i,)
            u = logitudinal_expansion(u,self.coarse_graining_half_spread)
            v = logitudinal_expansion(v,self.coarse_graining_half_spread)
            T = logitudinal_expansion(T,self.coarse_graining_half_spread)
            ugrid = logitudinal_expansion_dataset(ugrid,self.coarse_graining_half_spread)
            tgrid = logitudinal_expansion_dataset(tgrid,self.coarse_graining_half_spread)
        if maskpurposed:
            u = fillnarandom(u)
            v = fillnarandom(v)
            T = fillnarandom(T)

        pdi,_,_ = self.time_depth_indices(i)
        u0,v0,T0 = self.projections.project(u,prefix = 'u'),self.projections.project(v,prefix = 'u'),self.projections.project(T,prefix = 't')

        u_t,v_t = ugrid2tgrid(u,v,ugrid,tgrid)
        u0_t,v0_t = ugrid2tgrid(u0,v0,ugrid,tgrid)
        
        avgf = self.projections.avg_fields.isel(depth = [pdi]).load()
        
        avgf_dict = {}
        avgf_dict['u_res'] = avgf['avg_u_res'].rename(ulat = 'lat',ulon = 'lon')
        avgf_dict['v_res'] = avgf['avg_v_res'].rename(ulat = 'lat',ulon = 'lon')
        avgf_dict['u_res_t'] = avgf['avg_u_t_res'].rename(tlat = 'lat',tlon = 'lon')
        avgf_dict['v_res_t'] = avgf['avg_v_t_res'].rename(tlat = 'lat',tlon = 'lon')
        avgf_dict['T_res'] = avgf['avg_T_res'].rename(tlat = 'lat',tlon = 'lon')
        pastkeys = list(avgf_dict.keys())
        def depth_naming(key,i):
            return f"{key}_depth_{i}"
        def depth_reading(key):
            if '_depth_' in key:
                j0 = key.find('_depth_') 
                j = j0 + len('_depth_')
                depthind = int(key[j:])
                return key[:j0],depthind,depthvals[depthind]
            return key,None,None
        
        for key in pastkeys:
            val = avgf_dict[key]
            depthvals = val.depth.values
            for i in range(len(depthvals)):
                avgf_dict[depth_naming(key,i)] = val.isel(depth = i)
            avgf_dict.pop(key)
        for key,val in avgf_dict.items():
            avgf_dict[key] = logitudinal_expansion(val,self.coarse_graining_half_spread)


        if maskpurposed:
            for key,val in avgf_dict.items():
                avgf_dict[key] = fillnarandom(val)
        
        avgf_tgrid = {key:val for key,val in avgf_dict.items() if 'T_res' in key or '_t_' in key}
        avgf_ugrid = {key:val for key,val in avgf_dict.items() if not ('T_res' in key or '_t_' in key)}
        uvars = dict(u=u,v=v,u_0 = u0, v_0=v0,**avgf_ugrid)
        tvars = dict(u_t = u_t, v_t = v_t,u_0_t = u0_t,v_0_t=v0_t,T_0 = T0, T = T,**avgf_tgrid)

        uvars = {f'__{key}__':val for key,val in uvars.items()}
        tvars = {f'__{key}__':val for key,val in tvars.items()}
        cg = self.coarse_grain if not maskpurposed else self.dry_coarse_grain

        uhres,ulres,thres,tlres = hreslres(uvars,tvars,ugrid,tgrid,*cg)
        fields = 'u v T'.split()
        fields = [f'__{key}__' for key in fields]
        u,v,T = [ulres[key] for key in fields if key in ulres] + [tlres[key] for key in fields if key in tlres]
        u.name = 'u'
        v.name = 'v'
        T.name = 'T'
        u = u.rename(lat = 'ulat',lon = 'ulon')
        v = v.rename(lat = 'ulat',lon = 'ulon')
        T = T.rename(lat = 'tlat',lon = 'tlon')
        fields = xr.merge([u,v,T])
        if maskpurposed:
            uforcings = [dict(left='u v'.split(), right='u v'.split(), names = 'Su Sv'.split())]
            tforcings = [ dict(left='u_t v_t'.split(), right='T '.split(), names = 'ST '.split())]
        else:
            uforcings = [
                dict(left='u v'.split(), right='u v'.split(), names = 'Su Sv'.split()),
                dict(left = 'u_0 v_0'.split(),  right = 'u_0 v_0'.split(),names = 'Su00 Sv00'.split()),
            ]
            su_names = lambda x : [depth_naming(key,i) for key in f'Su{x} Sv{x}'.split()]
            st_names = lambda x : [depth_naming(key,i) for key in f'ST{x} '.split()]
            u0v0 = 'u_0 v_0'.split()
            for i in range(len(depthvals)):
                uv_names = [depth_naming(key,i) for key in 'u_res v_res'.split()]
                uforcings.append(dict(left = u0v0,  right = uv_names,names = su_names('01')))
                uforcings.append(dict(left = uv_names,  right = u0v0,names = su_names('10')))
                uforcings.append(dict(left = uv_names,  right = uv_names,names = su_names('11')))
            
            tforcings = [
                dict(left='u_t v_t'.split(), right='T '.split(), names = 'ST '.split()),
                dict(left = 'u_0_t v_0_t'.split(),  right = 'T_0 '.split(),names = 'ST00 '.split()),
            ]
            u0v0 = 'u_0_t v_0_t'.split()
            for i in range(len(depthvals)):
                uv_names = [depth_naming(key,i) for key in 'u_res_t v_res_t'.split()]
                tforcings.append(dict(left = u0v0,  right = uv_names,names = st_names('01')))
                tforcings.append(dict(left = uv_names,  right = u0v0,names = st_names('10')))
                tforcings.append(dict(left = uv_names,  right = uv_names,names = st_names('11')))
        def collect_keys(vardict,entries):
            keys = []
            for key in vardict:
                for entry in entries:
                    if f'__{entry}__' in key:
                        keys.append(key)
            return keys
        forcings = []
        for fd in uforcings:
            uhres_left = {name:uhres[key] for name,key in zip('u v'.split(),collect_keys(uhres,fd['left']))}
            uhres_right = {key:uhres[key] for key in collect_keys(uhres,fd['right'])}
            ulres_left =  {name:ulres[key] for name,key in zip('u v'.split(),collect_keys(ulres,fd['left']))}
            ulres_right = {key:ulres[key] for key in collect_keys(ulres,fd['right'])}
            root_names = [f'__{entry}__' for entry in fd['right']]
            target_names = fd['names']
            S = subgrid_forcing(uhres_left,ulres_left,uhres_right,ulres_right,root_names,target_names,cg[0])
            S = S.rename(lat = 'ulat',lon = 'ulon')
            forcings.append(S)


        for fd in tforcings:
            uhres_left = {name:thres[key] for name,key in zip('u v'.split(),collect_keys(thres,fd['left']))}
            uhres_right = {key:thres[key] for key in collect_keys(thres,fd['right'])}
            ulres_left =  {name:tlres[key] for name,key in zip('u v'.split(),collect_keys(tlres,fd['left']))}
            ulres_right = {key:tlres[key] for key in collect_keys(tlres,fd['right'])}
            root_names = [f'__{entry}__' for entry in fd['right']]
            target_names = fd['names']
            S = subgrid_forcing(uhres_left,ulres_left,uhres_right,ulres_right,root_names,target_names,cg[1])
            S = S.rename(lat = 'tlat',lon = 'tlon')
            forcings.append(S)

        forcings = xr.merge(forcings)
        
        if not maskpurposed:
            dropnames = []
            # newvars= {}
            for key in 'u v T'.split():
                for di in range(len(depthvals)):
                    extensions = '01 10 11'.split()
                    extensions = ['00'] + [depth_naming(ext,di) for ext in extensions]
                    fnames = [f'S{key}{ext}' for ext in extensions]
                    dropnames.extend(fnames)
                    forcings[f'S{key}0'] = forcings[fnames[0]]
                    forcings[depth_naming(f'S{key}1',di)] = forcings[fnames[0]] + forcings[fnames[1]] + forcings[fnames[2]] + forcings[fnames[3]]

            forcings = forcings.drop(dropnames)
            deep_train = []
            for i,key in enumerate(list(forcings.data_vars)):
                val = forcings[key].copy()
                nkey,di,dv=depth_reading(key)
                if dv is None:
                    continue
                val = val.expand_dims({'tr_depth':[dv]},axis =0 )
                val.name = nkey
                deep_train.append(val)
                forcings = forcings.drop(key)

            deep_train = xr.merge(deep_train)

            forcings = xr.merge([forcings,deep_train])
        outputs = xr.merge([fields,forcings])
        if maskpurposed:
            ugridmask = outputs.u*0
            tgridmask = outputs.T*0
            for name in 'u v T Su Sv ST'.split():
                vals = outputs[name].copy()
                vals = xr.where((np.abs(vals) + np.isnan(vals))>0,1,0)
                if 'u' in vals.dims[0]:
                    ugridmask = ugridmask + vals
                else:
                    tgridmask = tgridmask + vals

            ugridmask = xr.where( ((ugridmask>0)  + np.isnan(ugridmask)) > 0,0,1)
            tgridmask = xr.where( ((tgridmask>0)  + np.isnan(tgridmask) )> 0,0,1)

            ugridmask.name = 'ugrid_wetmask'
            tgridmask.name = 'tgrid_wetmask'
            outputs = xr.merge([ugridmask,tgridmask])
        outputs = trim_expanded_longitude(outputs,expansion = self.coarse_graining_crop)
        outputs = outputs.expand_dims(dim = {"depth": [depth]},axis=0)
        if not maskpurposed:
            outputs = outputs.expand_dims(dim = {"time": [str(time)]},axis=0)
        return outputs
    def is_wetmask_needed(self,i):
        if self.wetmask is None:
            return True
        _,di,_ = self.time_depth_indices(i)
        depth = self.ds.depth.values[di]
        deperr = np.amin(np.abs(self.wetmask.depth.values - depth))
        if deperr > 1e-1:
            return True
        return False
    def append2wetmask(self,wetmask):
        if self.wetmask is None:
            self.wetmask = wetmask
            return
        self.wetmask = xr.merge([self.wetmask,wetmask])        
    def __getitem__(self,i):
        ds = self.hres2lres(i,maskpurposed=False)
        if self.is_wetmask_needed(i):
            wetmask = self.hres2lres(i,maskpurposed=True)
            self.append2wetmask(wetmask)
        else:
            ds = xr.merge([self.wetmask,ds])
        # ds.to_netcdf('outputs.nc',mode = 'w')
        # raise Exception
        return tonumpydict(ds)
    def save_projections(self,):
        _,_,_,_,_,ugrid,tgrid = self.get_hres(0,fillna = True)

        ugrid = logitudinal_expansion_dataset(ugrid,self.coarse_graining_half_spread)
        tgrid = logitudinal_expansion_dataset(tgrid,self.coarse_graining_half_spread)
        
        projections = coarse_grain_inversion_weights(ugrid,tgrid,self.sigma)
        print(coarse_graining_projection_weights_path(self.sigma))
        projections.to_netcdf(coarse_graining_projection_weights_path(self.sigma),mode = 'w')

    def average_hres_fields(self,):
        t1 = int(np.floor(len(self.ds.time)*TIMES['train'][1]))
        avg_fields = self.ds.isel(time = 0).load()
        flushed_print(0)
        j = 1
        for i in range(1,t1):
            avg_fields +=  self.ds.isel(time = i).load()
            j+=1
            flushed_print(i)
        avg_fields = avg_fields/j
        return avg_fields
    def save_average_residual_fields(self,):
        filename = average_highres_fields_path(self.is_deep())
        avg_fields = xr.open_dataset(filename)
        u,v,T = avg_fields.u,avg_fields.v,avg_fields.T
        u = u.rename(ulat = "lat",ulon = "lon")
        v = v.rename(ulat = "lat",ulon = "lon")
        T = T.rename(tlat = "lat",tlon = "lon")
        u = logitudinal_expansion(u,self.coarse_graining_half_spread,stacked=self.is_deep())
        v = logitudinal_expansion(v,self.coarse_graining_half_spread,stacked=self.is_deep())
        T = logitudinal_expansion(T,self.coarse_graining_half_spread,stacked=self.is_deep())
        _,_,_,_,_,ugrid,tgrid = self.init_coarse_graining(0)
        u_t,v_t = ugrid2tgrid(u,v,ugrid,tgrid,stacked=self.is_deep())
        
        u1 = u - self.projections.project(u,prefix = 'u',stacked = self.is_deep())
        v1 = v - self.projections.project(v,prefix = 'u',stacked = self.is_deep())
        ut1 = u_t - self.projections.project(u_t,prefix = 't',stacked = self.is_deep())
        vt1 = v_t - self.projections.project(v_t,prefix = 't',stacked = self.is_deep())
        T1 = T - self.projections.project(T,prefix = 't',stacked = self.is_deep())
        u1 = u1.rename(lat = 'ulat',lon = 'ulon')
        v1 = v1.rename(lat = 'ulat',lon = 'ulon')
        ut1 = ut1.rename(lat = 'tlat',lon = 'tlon')
        vt1 = vt1.rename(lat = 'tlat',lon = 'tlon')
        T1 = T1.rename(lat = 'tlat',lon = 'tlon')

        k0s = list(avg_fields.data_vars.keys())
        avg_fields['avg_u_res'] = u1
        avg_fields['avg_v_res'] = v1
        avg_fields['avg_u_t_res'] = ut1
        avg_fields['avg_v_t_res'] = vt1
        avg_fields['avg_T_res'] = T1

        for k in k0s:
            avg_fields = avg_fields.drop(k)
        if 'time' in avg_fields.coords.keys():
            avg_fields = avg_fields.drop('time')

        filename = average_lowhres_fields_path(self.sigma,self.is_deep())
        avg_fields.to_netcdf(filename,mode = 'w')


            


