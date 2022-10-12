import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class ClimateNet(nn.Module):
    def __init__(self,spread=0,coarsen=0,rescale=[1/10,1/1e7],latsig=False,\
                 timeshuffle=True,direct_coord=True,longitude=False,latsign=False,gan=False):
        super(ClimateNet, self).__init__()
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.generative=False
        self.timeshuffle=timeshuffle
        self.device = torch.device(device)
        self.spread=spread
        self.latsig=latsig
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsign=latsign
        self.coarsen=coarsen
        self.coarse_grain_filters=[]
        self.coarse_grain_filters.append([])
        self.nn_layers = nn.ModuleList()
        self.init_coarsen=coarsen
        self.rescale=rescale
        self.gan=gan
        self.nprecision=0
        for m in range(1,9):
            gauss1=torch.zeros(2*m+1,2*m+1,dtype=torch.float32,requires_grad=False)
            for i in range(m):
                for j in range(m):
                    gauss1[i,j]=np.exp( -(j**2+i**2)/((2*m)**2)/2)
            gauss1=gauss1/gauss1.sum()
            self.coarse_grain_filters.append(torch.reshape(gauss1,[1,1,2*m+1,2*m+1]).to(device))
    def coarse_grain(self,x,m):
        if m==0:
            return x
        b=x.shape[0]
        c=x.shape[1]
        h=x.shape[2]
        w=x.shape[3]
        return F.conv2d(x.view(b*c,1,h,w),self.coarse_grain_filters[m]).view(b,c,h-2*m,w-2*m)
    def set_coarsening(self,c):
        c_=self.coarsen
        self.coarsen=c
        self.spread=self.spread-c_+c
    def initial_coarsening(self,):
        self.spread=self.spread+self.init_coarsen-self.coarsen
        self.coarsen=self.init_coarsen
