from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
class CNNLayer(nn.Module):
    def __init__(self,widthin,widthout, kernel, batchnorm,skip,nnlnr):
        super(CNNLayer,self).__init__()
        self.skip = skip
        self.kernel = nn.Conv2d(widthin,widthout,kernel)
        self.batchnorm  = None
        self.relu = None
        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(widthout)
        if nnlnr:
            self.relu = nn.ReLU(inplace = True)
    def subforward(self,x):
        x = self.kernel(x)
        for m in [self.batchnorm,self.relu]:
            if m is not None:
                x = self.batchnorm(x)
        return x
    def forward(self,x):
        if self.skip:
            return x + self.subforward(x)
        else:
            return self.subforward(x)


class CNN(nn.Module):
    def __init__(self,widths,kernels,batchnorm,skipconn,seed):#,**kwargs):
        super(CNN, self).__init__()
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = device

        self.skipcons = False
        layers = OrderedDict()
        spread = 0
        for i in range(len(kernels)):
            spread+=kernels[i]-1
        self.spread = spread//2

        torch.manual_seed(seed)


        lastlayer = lambda i: i != len(kernels) -1
        for i in range(len(kernels)):
            layers[f'conv-{i}'] = CNNLayer(widths[i],widths[i+1],kernels[i],\
                batchnorm[i],skipconn[i],lastlayer(i))

        for u in layers:
            layers[u] = layers[u].to(device)
        self.receptive_field=int(spread*2+1)
        self.conv_body = nn.Sequential(layers)
        softplus = OrderedDict()
        softplus['softplus'] = nn.Softplus()
        self.softplus = nn.Sequential(softplus)

        # initbatchnorm = OrderedDict()
        # initbatchnorm['batchnorm'] = nn.BatchNorm2d(widths[0])
        # self.initbatchnorm = nn.Sequential(initbatchnorm)


    def forward(self, x):
        # x = self.initbatchnorm(x)
        x = self.conv_body(x)
        mean,precision=torch.split(x,x.shape[1]//2,dim=1)
        precision=self.softplus(precision)
        return mean,precision


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def adjustcnn(widths,kernels,batchnorm,skipconn,seed,kernel_factor = 1.,width_factor =1., kernel_size = -1,constant_nparam = True):
    kernels = list(kernels)
    widths = list(widths)
    def compute_view_field(kernels):
        spread = 0
        for i in range(len(kernels)):
            spread+=kernels[i]-1
        return spread

    n0 = count_parameters(CNN(widths,kernels,batchnorm,skipconn,seed))
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
        n1 = count_parameters(CNN(widths,kernels,batchnorm,skipconn,seed))
        wd = np.array(widths[1:-1])
        wd = np.round(wd * np.sqrt(n0/n1) * width_factor).astype(int).tolist()
        widths = [widths[0]] + wd + [widths[-1]]
    return widths,kernels



class LCNN(nn.Module):
    def __init__(self,widths,kernels,batchnorm,skipconn,seed):#,**kwargs):
        super(CNN, self).__init__()
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = device

        self.skipcons = False
        # layers = OrderedDict()
        spread = 0
        for i in range(len(kernels)):
            spread+=(kernels[i]-1)/2
        spread = int(spread)

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



        # lastlayer = lambda i: i != len(kernels) -1
        # for i in range(len(kernels)):
        #     layers[f'conv-{i}'] = CNNLayer(widths[i],widths[i+1],kernels[i],\
        #         batchnorm[i],skipconn[i],lastlayer(i))
        #     # layers[f'conv-{i}'] = nn.Conv2d(widths[i],widths[i+1],kernels[i])
        #     # layers[f'norm-{i}'] = nn.BatchNorm2d(widths[i+1])
        #     # layers[f'relu-{i}'] = nn.ReLU(inplace = True)

        # for u in layers:
        #     layers[u] = layers[u].to(device)
        self.receptive_field=int(spread*2+1)
        # self.conv_body = nn.Sequential(layers)
        # softplus = OrderedDict()
        # softplus['softplus'] = nn.Softplus()
        # self.softplus = nn.Sequential(softplus)


    def forward(self, x):
        # x = self.conv_body(x)
        # mean,precision=torch.split(x,x.shape[1]//2,dim=1)
        # precision=self.softplus(precision)
        # return mean,precision
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
