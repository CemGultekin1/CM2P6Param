



from data.scalars import load_scalars
from models.load import load_model
from utils.arguments import options
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np



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
        out=torch.cat([mean,precision],dim=1)
        return out

def batch_normalization_weights(btchnrm,):
    d = btchnrm.weight.shape[0]
    x0 = torch.zeros(d,1).reshape([1,d,1,1])
    x = torch.eye(d).reshape([d,d,1,1])
    bias = btchnrm(x0).reshape(d,)
    weight = btchnrm(x).reshape(d,d) - bias.reshape(1,d)
    weight = torch.diag(weight)
    return weight,bias

def get_statedict_layer(st,lyr):
    keys = np.array(list(st.keys()))
    lyrnum = np.array([int(a.split('.')[1]) for a in keys])
    I = lyrnum == lyr
    keys = keys[I]
    return {k0:st[k0] for k0 in keys}

def rename_layer(d,lyr0,lyr1):
    keys = list(d.keys())
    for i in range(len(keys)):
        keys[i] = keys[i].replace(f'.{lyr0}.',f'.{lyr1}.')
    return {k:d[k0] for k,k0 in zip(keys, list(d.keys()))}
        
def match_layers(net0,net):
    lyrmatch = []
    j = 0
    for i in range(len(net0.nn_layers)):
        while type(net.nn_layers[j]) != type(net0.nn_layers[i]):
            j+=1
        lyrmatch.append((i,j))
    st = net.state_dict()
    st0 = net0.state_dict()
    for i,j in lyrmatch:
        d_ = get_statedict_layer(st,j)
        d_ = rename_layer(d_,j,i)
        for k,v in d_.items():
            st0[k] = v
    return st0





def main():
    path = '/scratch/cg3306/climate/CM2P6Param/jobs/cheng_train.txt'
    file1 = open(path, 'r')
    lines = file1.readlines()
    
    args = lines[0].split() + ['--mode','eval']
    runargs,_ = options(args,key = "run")
    _,state_dict,net,_,_,_,_,_=load_model(args)

    net0 = CNN()
    net0.eval()

    st0 = match_layers(net0,net)

    net0.load_state_dict(st0)
    net0.eval()

    x = torch.randn(1,2,31,31)
    y0p0 = net0.forward(x)
    y0,p0 = torch.split(y0p0,y0p0.shape[1]//2,dim = 1)

    net.load_state_dict(state_dict["best_model"])
    net.eval()

    y,p = net.forward(x)

    yerr = torch.sum(torch.square(y - y0)).item()
    perr = torch.sum(torch.square(p - p0)).item()
    ys2 = torch.sum(torch.square(y)).item()
    ps2 = torch.sum(torch.square(p)).item()
    print('errors:\t',yerr/ys2,perr/ps2)
    
    # Specify a path
    PATH = "cheng_global_model.pt"

    # Save
    torch.save(net0.state_dict(), PATH)
    

    scs = load_scalars(args)
    names = list(scs.data_vars.keys())
    names = [n for n in names if 'temp' in n]
    for n in names:
        scs = scs.drop(n)
    # scs.to_netcdf("cheng_global_model_scalars.nc")
    scalar_declare = ''
    for key in scs.data_vars.keys():
        scalar_declare += f"{key} = {scs[key].values[0].item()}\n"
    print(scalar_declare)



if __name__=='__main__':
    main()