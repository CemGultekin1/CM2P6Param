import itertools
import os
import sys
from data.exceptions import RequestDoesntExist
from run.eval import get_lsrp_modelid
import torch
from data.load import get_data
from models.load import load_model
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
from utils.paths import TIME_LAPSE
from utils.slurm import flushed_print
import numpy as np
from utils.xarray import concat, fromtensor, fromtorchdict, fromtorchdict2tensor
import xarray as xr

class CoordinateLocalizer:
    def get_local_ids(self,coord,field_coords):
        cdict = dict(lat = coord[0],lon = coord[1])
        ldict = dict(lat = None,lon = None)

        for key,val in cdict.items():
            i = np.argmin(np.abs(field_coords[key].numpy() - val))
            ldict[key] = i
        return ldict
    
    def get_localized(self,coord,spread,field_coords,*args):
        ldict1 = self.get_local_ids(coord,field_coords)
        if spread > 0:
            latsl,lonsl = [slice(ldict1[key] - spread,ldict1[key] + spread + 1) for key in 'lat lon'.split()]
        else:
            latsl,lonsl = [slice(ldict1[key] ,ldict1[key] + 1) for key in 'lat lon'.split()]
        
        newargs = []
        for arg in args:
            newarg = dict()
            for key,data_var in arg.items():
                dims,vals = data_var
                if len(vals.shape) == 2:
                    newarg[key] = (dims,vals[latsl,lonsl])
                else:
                    newarg[key] = (dims,vals)
            newargs.append(newarg)

        newcoords = dict()
        dslice = dict(lat = latsl,lon = lonsl)
        for key,val in field_coords.items():
            if key not in 'lat lon'.split():
                newcoords[key] = val
            else:
                newcoords[key] = val[dslice[key]]
        newargs = [newcoords] + newargs
        return tuple(newargs)


def main():
    args = sys.argv[1:]
    runargs,_ = options(args,key = "run")

    modelid,_,net,_,_,_,_,runargs=load_model(args)
    device = get_device()
    lsrp_flag, lsrpid = get_lsrp_modelid(args)
    if lsrp_flag:
        raise NotImplemented
    
    kwargs = dict(contained = '' if not lsrp_flag else 'res')
    assert runargs.mode == "eval"
    multidatargs = populate_data_options(args,non_static_params=['depth','co2'],domain = 'global')
    stats = None
    coords = [(30,-60),(-20,-104)]
    localizer = CoordinateLocalizer()
    for datargs in multidatargs:
        try:
            test_generator, = get_data(datargs,half_spread = net.spread, torch_flag = False, data_loaders = True,groups = ('train',))
        except RequestDoesntExist:
            print('data not found!')
            test_generator = None
        if test_generator is None:
            continue
        nt = 0
        for fields,forcings,forcing_mask,field_coords,forcing_coords in test_generator:
            for cid,coord in  enumerate(coords):
                _,loc_fields, = localizer.get_localized(coord, net.spread,field_coords,fields, )
                loc_forcing_coords,loc_forcings,loc_forcing_mask, = localizer.get_localized(coord, 0,forcing_coords,forcings, forcing_mask)
  
                fields_tensor = fromtorchdict2tensor(loc_fields).type(torch.float32)
                depth = forcing_coords['depth'].item()
                co2 = forcing_coords['co2'].item()
                time = forcing_coords['time'].item()
                kwargs = dict(contained = '', \
                    expand_dims = {'co2':[co2],'depth':[depth],'time' : [time],'coord_id' :[cid]},\
                    drop_normalization = True,
                    )


                with torch.set_grad_enabled(False):
                    mean,std =  net.forward(fields_tensor.to(device))
                    mean = mean.to("cpu")
                    std = std.to("cpu")
                    std = torch.sqrt(1/std)
                


                predicted_forcings = fromtensor(mean,loc_forcings,loc_forcing_coords, loc_forcing_mask,denormalize = True,**kwargs)
                predicted_std = fromtensor(std,loc_forcings,loc_forcing_coords, loc_forcing_mask,denormalize = True,**kwargs)
                true_forcings = fromtorchdict(loc_forcings,loc_forcing_coords,loc_forcing_mask,denormalize = True,**kwargs)
                
                output_dict = dict(
                    predicted_forcings=('pred_', predicted_forcings,'_mean'),
                    predicted_std = ('pred_',predicted_std,'_std'),
                    true_forcings = ('true_',true_forcings,'')
                )
                data_vars = {}
                for key,val in output_dict.items():
                    pref,vals,suff = val
                    names = list(vals.data_vars.keys())
                    rename_dict = {nm : pref +nm + suff  for nm in names}
                    vals = vals.rename(rename_dict).isel(lat = 0,lon = 0).drop('lat lon'.split())
                    for name in rename_dict.values():
                        data_vars[name] = vals[name]
                data_vars['lat'] = xr.DataArray(data = [coord[0]],dims = ['coord_id'],coords = dict(
                    coord_id = (['coord_id'],[cid])
                ) )
                data_vars['lon'] = xr.DataArray(data = [coord[1]],dims = ['coord_id'],coords = dict(
                    coord_id = (['coord_id'],[cid])
                ) )
                ds = concat(**data_vars)
                if stats is None:
                    stats = ds
                else:
                    stats = xr.merge([stats,ds])
            nt += 1
            if nt == 400:
                break
            flushed_print('\t\t',nt)
    if not os.path.exists(TIME_LAPSE):
        os.makedirs(TIME_LAPSE)
    filename = os.path.join(TIME_LAPSE,modelid+'.nc')
    print(filename)
    stats.to_netcdf(filename,mode = 'w')


            

            






if __name__=='__main__':
    main()