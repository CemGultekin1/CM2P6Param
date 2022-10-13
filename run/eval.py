import sys
import torch
from data.load import get_data
from models.load import load_model
import matplotlib.pyplot as plt
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
from utils.xarray import numpydict2dataset

def preprocess(fields,forcings,masks,device):
    def value_norms(fields):
        return {key: fields[key]['val']  for key in fields},{key: fields[key]['normalization']  for key in fields}
    def normalize(fields,nfields):
        def _norm(f,n):
            # print('f.shape,n.shape',f.shape,n.shape)
            return (f - n[:,0].reshape([-1,1,1]))/n[:,1].reshape([-1,1,1])
        return {key: _norm(fields[key],nfields[key])  for key in fields}
    def torch_stack(fields):
        fv = list(fields.values())
        return torch.stack(fv,dim = 1).type(torch.float32).to(device)
    net_fields = torch_stack(normalize(*value_norms(fields)))
    net_forcings = torch_stack(normalize(*value_norms(forcings)))
    net_masks = torch_stack(value_norms(masks)[0])
    return net_fields,net_forcings,net_masks


def postprocessing(forcings,masks,outputs,):
    outputs = torch.unbind(outputs.to('cpu'), dim = 1)
    def value_norms(fields):
        return {key: fields[key]['val']  for key in fields},{key: fields[key]['normalization']  for key in fields}

    def denormalize(fields,nfields):
        def _denorm(f,n):
            return f#*n[:,1].reshape([-1,1,1]) + n[:,0].reshape([-1,1,1])
        return {f"mean_{key}": {'val' : _denorm(fields[i],nfields[key])}  for i,key in enumerate(nfields.keys())}
    _,frcnorms = value_norms(forcings)
    outputs = denormalize(outputs,frcnorms)
    for key in forcings:
        for key_ in forcings[key]:
            if key_ != 'val':
                outputs[f"mean_{key}"][key_] = forcings[key][key_]

        outputs[f"mean_{key}"]['val'][~masks[key]['val']] = 0#np.nan
        forcings[key]['val'][~masks[key]['val']] = 0#np.nan
    return dict(forcings,**outputs)

def concat_datasets(x,y):
    for key in y:
        for key_ in y[key]:
            v0 = x[key][key_]
            v1 = y[key][key_]
            x[key][key_] = torch.cat((v0,v1),dim = 0)
    return x


def main():
    args = sys.argv[1:]
    multidatargs = populate_data_options(args,)
    for datargs in multidatargs:
        print(datargs)
        test_generator, = get_data(datargs,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('test',))
        dataprms, _ = options(datargs,key = "data")
        linsupres = dataprms.linsupres
        for outputs in test_generator:
            if linsupres:
                fields,forcings,lsrp_res_forcings,forcing_masks = outputs
            else:
                fields,forcings,forcing_masks = outputs
            print(fields)
            return
            net_fields,net_forcings,net_masks = preprocess(fields,forcings,masks,device)

            with torch.set_grad_enabled(False):
                mean,_ = net.forward(net_fields)

            if dataset is None:
                dataset = postprocessing(forcings,masks,mean,)
                localdataset = numpydict2dataset(dataset,time = i//10)
            else:
                dataset_ = postprocessing(forcings,masks,mean,)
                localdataset = numpydict2dataset(dataset_,time = i//10)
                dataset = concat_datasets(dataset,dataset_)
            i+=1
            print(i)
            n = 12
            if i<n:
                fig,axs = plt.subplots(1,2,figsize = (30,15))
                localdataset.isel(time =0 ).forcings_u.plot(ax = axs[0])
                localdataset.isel(time =0 ).mean_forcings_u.plot(ax = axs[1])
                fig.savefig(f'snapshot_local_time_{i}_new_1.png')
                plt.close()
            elif i==n:
                fulldataset = numpydict2dataset(dataset,)
                fig,axs = plt.subplots(1,2,figsize = (30,15))
                fulldataset.isel(time =0 ).forcings_u.plot(ax = axs[0])
                fulldataset.isel(time =0 ).mean_forcings_u.plot(ax = axs[1])
                fig.savefig('snapshot_global_time_0_new_1.png')
                plt.close()
                break





if __name__=='__main__':
    main()
