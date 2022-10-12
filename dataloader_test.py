
# def preprocess(fields,forcings,masks,device):
#     def value_norms(fields):
#         return {key: fields[key]['val']  for key in fields},{key: fields[key]['normalization']  for key in fields}
#     def normalize(fields,nfields):
#         def _norm(f,n):
#             # print('f.shape,n.shape',f.shape,n.shape)
#             return (f - n[:,0].reshape([-1,1,1]))/n[:,1].reshape([-1,1,1])
#         return {key: _norm(fields[key],nfields[key])  for key in fields}
#     def torch_stack(fields):
#         fv = list(fields.values())
#         return torch.stack(fv,dim = 1).type(torch.float32).to(device)
#     net_fields = torch_stack(normalize(*value_norms(fields)))
#     net_forcings = torch_stack(normalize(*value_norms(forcings)))
#     net_masks = torch_stack(value_norms(masks)[0])
#     return net_fields,net_forcings,net_masks


# def postprocessing(forcings,masks,outputs,):
#     outputs = torch.unbind(outputs.to('cpu'), dim = 1)
#     def value_norms(fields):
#         return {key: fields[key]['val']  for key in fields},{key: fields[key]['normalization']  for key in fields}
    
#     def denormalize(fields,nfields):
#         def _denorm(f,n):
#             return f#*n[:,1].reshape([-1,1,1]) + n[:,0].reshape([-1,1,1])
#         return {f"mean_{key}": {'val' : _denorm(fields[i],nfields[key])}  for i,key in enumerate(nfields.keys())}
#     _,frcnorms = value_norms(forcings)
#     outputs = denormalize(outputs,frcnorms)
#     for key in forcings:
#         for key_ in forcings[key]:
#             if key_ != 'val':
#                 outputs[f"mean_{key}"][key_] = forcings[key][key_]

#         outputs[f"mean_{key}"]['val'][~masks[key]['val']] = 0#np.nan
#         forcings[key]['val'][~masks[key]['val']] = 0#np.nan
#     return dict(forcings,**outputs)

# def concat_datasets(x,y):
#     for key in y:
#         for key_ in y[key]:
#             v0 = x[key][key_]
#             v1 = y[key][key_]
#             x[key][key_] = torch.cat((v0,v1),dim = 0)
#     return x

# def to_numpy(*torchdicts):
#     torchdicts = list(torchdicts)
#     for i in range(len(torchdicts)):
#         for key in torchdicts[i]:
#             for key_ in torchdicts[i][key]:
#                 torchdicts[i][key][key_] = torchdicts[i][key][key_].numpy()


# def to_xarray(fields):
#     fields_ = []
#     for name,fieldlist in fields.items():
#         for i,(field,lat,lon) in enumerate(zip(fieldlist['val'],fieldlist['lat'],fieldlist['lon'])):
#             fields_.append(xr.DataArray(
#                 data = field,
#                 dims = ["lat","lon"],
#                 name = name,
#                 coords = dict(lat = lat,lon = lon)
#             ))

#     return xr.merge(fields_)
# def concat(fields,mask):
#     for name,field in fields.items():
#         for i,(f,m) in enumerate(zip(field['val'],mask[f"{name}_mask"]['val'])):
#             f[~m] = np.nan
#             fields[name]['val'][i] = f
#     fields = to_xarray(fields)
#     return fields



# import sys
import itertools
from data.load import get_data
from models.load import load_model
from utils.parallel import get_device
from utils.slurm import flushed_print
import torch

def main():
    args = "--sigma 4 --domain global --depth 5 --parts 3 3 --minibatch 1 --prefetch_factor 2 --linsupres True --temperature True --latitude True --num_workers 1 --mode train".split()
    generator,= get_data(args,half_spread = 10, torch_flag = True, data_loaders = True,groups = ('all',))

    modelid,state_dict,net,criterion,optimizer,scheduler,logs,runargs=load_model(args)

    flushed_print(runargs)
    training_generator,val_generator=get_data(args,half_spread = 0,torch_flag = True,data_loaders = True,groups = ('train','validation'))
    device=get_device()
    # flushed_print("epochs started")
    # timer = Timer()

    import matplotlib.pyplot as plt 
    import numpy as np
    for epoch in range(runargs.epoch,runargs.maxepoch):
        for infields,outfields,mask in training_generator: 
            outfields[mask<1] = np.nan
            vars = [infields,outfields,mask]
            vars = [var[0].numpy() for var in vars]
            # for var in vars:
            #     for var_ in var:
            #         v = var_[var_==var_]

            #         mom1 = np.mean(v)
            #         mom2 = np.sqrt(np.mean((v - mom1)**2))
            #         print(mom1,mom2)
            ncol,nrow = 3,3
            fig,axs = plt.subplots(nrow,ncol,figsize = (25,20))
            for i,j in itertools.product(range(nrow),range(ncol)):
                axs[i,j].imshow(vars[i][j])
            fig.savefig(f'dummy-{t}.png')
                
            # if not torch.any(mask):
            #     continue

            # timer.end('data')
            # infields,outfields,mask = preprocess(infields,outfields,mask,device,runargs.linsupres)
            # if isinstance(outfields,tuple):
            #     outfields = outfields[1]
            
            
            
            # timer.start('model')
            # mean,prec = net.forward(infields) 
            # print(outfields.shape,mean.shape,prec.shape)
            t+=1
            if t == 9 :
                return

if __name__=='__main__':
    main()