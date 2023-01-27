#!/bin/env python

from data.load import get_data
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import math
import xarray as xr
import time

from utils.xarray import fromtensor, fromtorchdict2dataset, fromtorchdict2tensor, normalize_dataset, plot_ds, skipna_mean

# GPU setup
args_no_cuda = False #True when manually turn off cuda
use_cuda = not args_no_cuda and torch.cuda.is_available()
if use_cuda:
    print('device for inference on',torch.cuda.device_count(),'GPU(s)')
else:
    print('device for inference on CPU')



#load the neural network
class CNN(nn.Module):
    def __init__(self,filter_size=[5, 5, 3, 3, 3, 3, 3, 3],\
                     width=[128, 64, 32, 32, 32, 32, 32, 4],\
                        inchan=2,cuda_flag=False):
        super(CNN, self).__init__()
        self.nn_layers = nn.ModuleList()
        self.filter_size=filter_size
        self.num_layers=len(filter_size)
        
        if cuda_flag:
            device = "cuda:0" 
        else:  
            device = "cpu"  
        
        self.nn_layers.append(nn.Conv2d(inchan, width[0], filter_size[0]).to(device) )
        for i in range(1,self.num_layers):
            self.nn_layers.append(nn.BatchNorm2d(width[i-1]).to(device) )
            self.nn_layers.append(nn.Conv2d(width[i-1], width[i], filter_size[i]).to(device) )
        self.nn_layers.append(nn.Softplus().to(device))
    def forward(self, x):
        cn=0
        while cn<len(self.nn_layers)-2:
            x = self.nn_layers[cn](x)
            cn+=1
            x = F.relu(self.nn_layers[cn](x))
            cn+=1
        x=self.nn_layers[cn](x)
        mean,precision=torch.split(x,x.shape[1]//2,dim=1)
        precision=self.nn_layers[-1](precision)
        return mean,precision
        # out=torch.cat([mean,precision],dim=1)
        # return out

class ScaledCNN:
    def __init__(self,nn_load_file,input_scalars,output_scalars,name) -> None:
        print(f'{name}\tinput_scales:{input_scalars}\n{name}\toutput_scales:{output_scalars}')
        nn=CNN(cuda_flag=False)
        nn.load_state_dict(torch.load(nn_load_file,map_location='cpu'))
        nn.eval()
        self.input_scalars = input_scalars.reshape([1,-1,1,1])
        self.output_scalars = output_scalars.reshape([1,-1,1,1])
        self.cnn = nn
        one_output_mean,one_output_std = self.forward(torch.ones([1,2,30,30]))
        zero_output_mean,zero_output_std = self.forward(torch.zeros([1,2,30,30]))
        for key,i in zip("Su Sv".split(),range(2)):
            print(f'{name}\tone_input_{key}_mean:{one_output_mean[0,i].mean().item()}\tone_input_{key}_std:{one_output_std[0,i].mean().item()}')
            print(f'{name}\tzero_output_{key}_mean:{zero_output_mean[0,i].mean().item()}\tzero_input_{key}_std:{zero_output_std[0,i].mean().item()}')
        print()
    def forward(self,x):
        x = x/self.input_scalars
        x = x.type(torch.float32)
        with torch.no_grad():
            mean,precision = self.cnn.forward(x)
        mean = mean*self.output_scalars
        std = torch.sqrt(1/precision)*self.output_scalars
        std = std.type(torch.float64)
        mean = mean.type(torch.float64)
        return mean,std

def load_cheng_model():
    nn_load_file='cheng_global_model.pt'
    u_scale = 0.09439346225350978
    v_scale = 0.07252696573672539
    Su_scale = 4.9041400042653195e-08
    Sv_scale = 4.8550991806254025e-08
    input_scalars = np.array([u_scale,v_scale])
    output_scalars = np.array([Su_scale,Sv_scale])
    return ScaledCNN(nn_load_file, input_scalars,output_scalars,'V1')

def load_cheng_model_first_version():
    nn_load_file='v1-best-model.pt'
    u_scale=1/0.10278768092393875
    v_scale=1/0.07726840674877167
    world_radius_in_meters=6.371e6
    angle_to_meters=world_radius_in_meters*2*np.pi/360
    Su_scale=0.004745704121887684/angle_to_meters
    Sv_scale=0.004386111628264189/angle_to_meters
    input_scalars = 1/np.array([u_scale,v_scale])
    output_scalars = np.array([Su_scale,Sv_scale])
    return ScaledCNN(nn_load_file, input_scalars,output_scalars,'V0')

def main():
    cnn0 = load_cheng_model_first_version()
    cnn1 = load_cheng_model()
    # return
    args = '--sigma 4 --mode eval --num_workers 1'.split()
    test_generator, = get_data(args,half_spread = 10, torch_flag = False, data_loaders = True,groups = ('test',))
    kwargs = {}
    plot_regions = dict(
        glbl = dict(),
        loc1 = dict(lat = slice(-20,10),lon = slice(-150,-100)),
        loc2 = dict(lat = slice(-40,-20),lon = slice(50,80)),
        loc3 = dict(lat = slice(30,60),lon = slice(-30,0)),
        loc4 = dict(lat = slice(10,40),lon = slice(150,180)),
    )
    for _fields,_forcings,forcing_mask,field_coords,forcing_coords in test_generator:
        fields = fromtorchdict2dataset(_fields,field_coords)
        forcings = fromtorchdict2dataset(_forcings,forcing_coords)
        fields = normalize_dataset(fields,denormalize=True,drop_normalization=True)
        # for name,reg in plot_regions.items():
        #     fields_ = fields.sel(**reg)
        #     plot_ds(fields_,f'saves/plots/cheng_model_comparison/inputs-{name}',ncols = 2)
        # return

        forcings = normalize_dataset(forcings,denormalize=True,drop_normalization=True).fillna(0)
        
        # fields = fields.isel(lat = slice(250,410),lon = slice(250,410))
        # forcings = forcings.isel(lat = slice(260,400),lon = slice(260,400))
        uv = np.stack([np.stack([fields.u.values,fields.v.values])])
        uv = torch.from_numpy(uv)

        Suv = np.stack([np.stack([forcings.Su.values,forcings.Sv.values])])
        Suv = torch.from_numpy(Suv)
        print(uv.shape,Suv.shape)
        mean0,std0 = cnn0.forward(uv)
        mean0 = fromtensor(mean0,_forcings,forcing_coords, forcing_mask,denormalize = False,**kwargs)
        std0 = fromtensor(std0,_forcings,forcing_coords, forcing_mask,denormalize = False,**kwargs)

        mean1,std1 = cnn1.forward(uv)
        mean1 = fromtensor(mean1,_forcings,forcing_coords, forcing_mask,denormalize = False,**kwargs)
        std1 = fromtensor(std1,_forcings,forcing_coords, forcing_mask,denormalize = False,**kwargs)
    
        def rename(ds,suffix):
            keys = list(ds.data_vars.keys())
            newkeys = [f"{k}_{suffix}" for k in keys]
            return ds.rename({k0:k1 for k0,k1 in zip(keys,newkeys)})

        
    
        # scale_grouping = lambda key :[
        #     (False,[f"{key}err0",f"{key}err1",f"{key}meandiff"]),
        #     (False,[f"{key}std0",f"{key}std1",f"{key}stddiff"]),
        # ]
        scale_grouping = lambda key :[
            (False,[f"{key}mean0",f"{key}mean1"]),
            (False,[f"{key}err0",f"{key}err1"]),
        ]
        err0 = forcings - mean0
        err1 = forcings - mean1
        outputs = dict(
            mean0 = mean0,
            mean1 = mean1,
            err0 = err0,
            err1 = err1,
            diff = mean1 - mean0
        )
        for key,val in outputs.items():
            outputs[key] = rename(val,key)
        outputs = xr.merge(list(outputs.values()))
        
        for fname in 'Su Sv'.split():
            outputs_ = outputs.copy()
            keys = outputs.data_vars.keys()
            dropkeys = [k for k in keys if fname not in k]
            for dk in dropkeys:
                outputs_ = outputs_.drop(dk)
            for name,reg in plot_regions.items():
                plot_fields_ = outputs_.sel(**reg)
                plot_ds(plot_fields_,f'saves/plots/cheng_model_comparison/one-{fname}-{name}',ncols = 3,scale_grouping = scale_grouping(f'{fname}_'))
        return
        fields_tensor = fromtorchdict2tensor(fields).type(torch.float32)
        forcings = fromtensor(mean,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
        
        kwargs = dict(contained = '' if not lsrp_flag else 'res', \
            expand_dims = {'co2':[co2],'depth':[depth]},\
            drop_normalization = True,
            )
        if nt ==  0:
            flushed_print(depth,co2)

        with torch.set_grad_enabled(False):
            mean,_ =  net.forward(fields_tensor.to(device))
            mean = mean.to("cpu")


        predicted_forcings = fromtensor(mean,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
        true_forcings = fromtorchdict(forcings,forcing_coords,forcing_mask,denormalize = True,**kwargs)
    



if __name__ == '__main__':
    main()
    

def MOM6_testNN(u,v,pe,pe_num):
   global nn,gpu_id,u_scale,v_scale,Su_scale,Sv_scale
   # start_time = time.time()
   # print('PE number is',pe_num)
   # print('PE is',pe)
   # print(u.shape,v.shape)
   #normalize the input by training scaling
   u= u/u_scale
   v= v/v_scale
   x = np.array([np.squeeze(u),np.squeeze(v)])
   if x.ndim==3:
     x = x[:,:,:,np.newaxis]
   x = x.astype(np.float32)
   x = x.transpose((3,0,1,2)) # new the shape is (nk,2,ni,nj)
   x = torch.from_numpy(x) # quite faster than x = torch.tensor(x)
   if use_cuda:
       if not next(nn.parameters()).is_cuda:
          gpu_id = int(pe/math.ceil(pe_num/torch.cuda.device_count()))
          print('GPU id is:',gpu_id)
          nn = nn.cuda(gpu_id)
       x = x.cuda(gpu_id)
   with torch.no_grad():
       # start_time = time.time()
       out = nn.forward(x)
       # end_time = time.time()
   if use_cuda:
       out = out.to('cpu')
   out = out.numpy().astype(np.float64)
   # At this point, python out shape is (nk,4,ni,nj)
   # Comment-out is tranferring arraies into F order
   """
   print(out.shape)
   dim = np.shape(out)
   out = out.flatten(order='F')
   out = out.reshape(dim[0],dim[1],dim[2],dim[3], order='F')
   """
   # convert out to (ni,nj,nk)
   out = out.transpose((1,2,3,0)) # new the shape is (4,ni,nj,nk)
   dim = np.shape(out)
   # print(dim)
   Sxy = np.zeros((2,dim[1],dim[2],dim[3])) # the shape is (2,ni,nj,nk)
   epsilon_x = np.random.normal(0, 1, size=(dim[1],dim[2]))
   epsilon_x = np.dstack([epsilon_x]*dim[3])
   epsilon_y = np.random.normal(0, 1, size=(dim[1],dim[2]))
   epsilon_y = np.dstack([epsilon_y]*dim[3])
   # if pe==0:
   #   print(scaling)
   """
   # mean output
   Sxy[0,:,:,:] = (out[0,:,:,:])*Su_scale
   Sxy[1,:,:,:] = (out[1,:,:,:])*Sv_scale
   # std output
   Sxy[0,:,:,:] = (epsilon_x/out[2,:,:,:])*Su_scale
   Sxy[1,:,:,:] = (epsilon_y/out[3,:,:,:])*Sv_scale
   """
   # full output
   Sxy[0,:,:,:] = (out[0,:,:,:] + epsilon_x*np.sqrt(1/out[2,:,:,:]))*Su_scale
   Sxy[1,:,:,:] = (out[1,:,:,:] + epsilon_y*np.sqrt(1/out[3,:,:,:]))*Sv_scale
   """
   # scaling the parameters for upper and lower layers
   Sxy[:,:,:,0]=Sxy[:,:,:,0]*0.8
   Sxy[:,:,:,1]=Sxy[:,:,:,1]*1.5
   """
   """
   np.savetxt('Sx_mean.txt',out[0,:,:,0])
   np.savetxt('Sx_std.txt',out[2,:,:,0])
   np.savetxt('WH_u.txt',u[:,:,1])
   np.savetxt('Sx.txt',Sxy[0,:,:,0])
   """
   # end_time = time.time()
   # print("--- %s seconds for CNN ---" % (end_time - start_time))
   # print(nn)
   # print(Sxy.shape)
   return Sxy 