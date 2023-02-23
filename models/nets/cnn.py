from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils.parallel import get_device

class Layer:
    def __init__(self,nn_layers:list,device) -> None:
        self.device = device
        self.nn_layers =nn_layers
        self.section = []
    def add(self,nn_obj):
        self.section.append(len(self.nn_layers))
        self.nn_layers.append(nn_obj.to(self.device))
    def __call__(self,x):
        for j in self.section:
            x = self.nn_layers[j](x)
        return x

class CNN_Layer(Layer):
    def __init__(self,nn_layers:list,device,widthin,widthout,kernel,batchnorm,nnlnr) -> None:
        super().__init__(nn_layers,device)
        self.add(nn.Conv2d(widthin,widthout,kernel))
        if batchnorm:
            self.add(nn.BatchNorm2d(widthout))
        if nnlnr:
            self.add(nn.ReLU(inplace = True))
class Softmax_Layer(Layer):
    def __init__(self,nn_layers:list,device,split) -> None:
        super().__init__(nn_layers,device)
        self.add(nn.Softplus())
        self.split = split
    def __call__(self, x):
        if self.split>1:
            xs = list(torch.split(x,x.shape[1]//self.split,dim=1))
            xs[-1] = super().__call__(xs[-1])
            return tuple(xs)
        return super().__call__(x)
        

class Sequential(Layer):
    def __init__(self,nn_layers,device,widths,kernels,batchnorm,softmax_layer = False,split = 1):
        super().__init__(nn_layers,device)
        self.sections = []
        spread = 0
        self.nlayers = len(kernels)
        for i in range(self.nlayers):
            spread+=kernels[i]-1
        self.spread = spread//2
        for i in range(self.nlayers):
            self.sections.append(CNN_Layer(nn_layers,device,widths[i],widths[i+1],kernels[i],batchnorm[i], i < self.nlayers - 1))
        if softmax_layer:
            self.sections.append(Softmax_Layer(nn_layers,device,split))
    def __call__(self, x):
        for lyr in self.sections:
            x = lyr.__call__(x)
        return x

class CNN(nn.Module):
    def __init__(self,widths = None,kernels = None,batchnorm = None,seed = None,**kwargs):
        super(CNN, self).__init__()
        device = get_device()
        self.device = device
        torch.manual_seed(seed)
        self.nn_layers = nn.ModuleList()
        # self.sequence :Dict[str,Layer] = dict()
        
        self.sequence = \
            Sequential(self.nn_layers,device, widths,kernels,batchnorm,softmax_layer=True,split = 2)
        # self.actives = list(self.sequence.keys())

        spread = 0
        for k in kernels:
            spread += (k-1)/2
        self.spread = int(spread)
    # def set_actives(self,sts:list):
    #     actives = list(self.sequence.keys())
    #     new_actives = []
    #     for st in sts:
    #         if isinstance(st,int):
    #             st = actives[st]
    #         assert st in self.sequence
    #         new_actives.append(st)
    #     self.actives = new_actives
    # def copy_if_needed(self,x,i):
    #     if len(self.sequence)>1:
    #         return torch.clone(x)
    #     return x
    def forward(self,x):
        x1 = x.to(self.device)
        x1 = self.sequence(x1)
        return x1
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def adjustcnn(kernel_factor = 1.,width_factor =1., kernel_size = -1,constant_nparam = True,**kwargs):
    kernels = kwargs.pop('kernels')
    widths = kwargs.pop('widths')
    kernels = list(kernels)
    widths = list(widths)
    def compute_view_field(kernels):
        spread = 0
        for i in range(len(kernels)):
            spread+=kernels[i]-1
        return spread

    n0 = count_parameters(CNN(kernels = kernels,widths = widths,**kwargs))
    view_field = compute_view_field(kernels)
    def compare(view,):
        if kernel_size < 0 :
            return view > view_field*kernel_factor
        return view > kernel_size - 1
    i=0
    while compare(compute_view_field(kernels)):
        K = np.amax(np.array(kernels))
        if K ==1 :
            break
        I = np.where(np.array(kernels)== K)[0]
        i = I[-1]
        kernels[i]-=1
    if compute_view_field(kernels)%2 == 1 :
        kernels[i]+=1
    if constant_nparam:
        n1 = count_parameters(CNN(kernels = kernels,widths = widths,**kwargs))
        wd = np.array(widths[1:-1])
        wd = np.round(wd * np.sqrt(n0/n1) * width_factor).astype(int).tolist()
        widths = [widths[0]] + wd + [widths[-1]]
    return widths,kernels

def kernels2spread(kernels):
    spread = 0
    for i in range(len(kernels)):
        spread+=(kernels[i]-1)/2
    spread = int(spread)
    return spread




class LCNN(nn.Module):
    def __init__(self,widths,kernels,batchnorm,skipconn,seed):#,**kwargs):
        super(LCNN, self).__init__()
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = device

        self.skipcons = False
        # layers = OrderedDict()
        
        self.spread = kernels2spread(kernels)
        torch.manual_seed(seed)

        self.nn_layers = nn.ModuleList()
        initwidth = widths[0]
        width = widths[1:]

        filter_size = kernels
        self.nn_layers.append(nn.Conv2d(initwidth, width[0], filter_size[0]) )
        self.num_layers = len(filter_size)
        for i in range(1,self.num_layers):
            self.nn_layers.append(nn.BatchNorm2d(width[i-1]).to(device) )
            self.nn_layers.append(nn.Conv2d(width[i-1], width[i], filter_size[i]) )
        self.nn_layers.append(nn.Softplus())
        self.receptive_field=int(self.spread*2+1)


    def forward(self, x):
        cn=0
        for _ in range(self.num_layers-1):
            x = self.nn_layers[cn](x)
            cn+=1
            x = F.relu(self.nn_layers[cn](x))
            cn+=1
        x=self.nn_layers[cn](x)
        cn+=1
        mean,precision=torch.split(x,x.shape[1]//2,dim=1)
        precision=self.nn_layers[cn](precision)

        return mean,precision



class DoubleLCNNWrapper(LCNN):
    var_net :LCNN
    def add_var_part(self,net:LCNN):
        self.var_net = net
    def forward(self, x):
        mean,_ =  super().forward(x)
        _,var = self.var_net.forward(x)
        return mean,1/var
    def train(self,mean = True,var = True):
        if mean:
            super().train()
        if var:
            self.var_net.train()
    def eval(self,mean = True,var = True):
        if mean:
            super().eval()
        if var:
            self.var_net.eval()