from itertools import combinations_with_replacement

import torch
from torch import nn



class RegressionModel:
    def __init__(self,spread=1, degree=3,initwidth=3,outwidth=3,latsig=False,                    latsign=False,                    direct_coord=False,                    freq_coord=False):
        super(RegressionModel, self).__init__()
        device=self.device
        self.nn_layers = nn.ModuleList()
        self.spread=spread
        self.nparam=0
        self.freq_coord=freq_coord
        self.heteroscedastic=False
        self.direct_coord=direct_coord
        self.longitude=False
        self.latsign=latsign
        self.latsig=latsig
        self.degree=degree
        receptive_field=2*spread+1
        self.receptive_field=receptive_field
        if self.direct_coord:
            if self.latsig:
                initwidth+=1
            if self.latsign:
                initwidth+=1
            if self.longitude:
                initwidth+=1
        elif self.freq_coord:
            if self.latsig:
                initwidth+=1
            if self.latsign:
                initwidth+=1
            if self.longitude:
                initwidth+=2
        self.initwidth=initwidth
        self.outwidth=outwidth
        featurenum=initwidth*receptive_field**2
        self.featurenum=featurenum
        T=nn.Conv2d(initwidth, featurenum+1, 2*spread+1)
        W=T.weight.data
        W=W*0
        W.requires_grad=False
        names=[]
        for i in range(initwidth):
            for j in range(featurenum):
                j_=j
                j1=j_%receptive_field
                j_=j_//receptive_field
                j2=j_%receptive_field
                j_=j_//receptive_field
                j3=j_%initwidth
                W_=torch.zeros(receptive_field,receptive_field)
                W_[j1,j2]=1
                W[j,i]=W_.view(1,1,receptive_field,receptive_field)
                names.append([j3,[j1-spread,j2-spread]])
        self.basic_names=names.copy()
        T.weight.data=W
        T.bias.data=T.bias.data*0
        for j in range(featurenum,featurenum+1):
            T.bias.data[j]=1.
        T.bias.data.requires_grad=False
        self.nn_layers.append(T.to(device) )
        N=outwidth+1
        self.res = list(combinations_with_replacement(range(N), degree))
        self.res = [torch.tensor(I) for I in self.res]
        self.names=[]
        self.outputdimen=len(self.res)
    def compute_names(self,):
        outwidth=self.outwidth
        self.names=[]
        names=self.basic_names.copy()
        featurenum=self.featurenum
        for i in range(len(self.res)):
            I=self.res[i]
            D=torch.zeros(featurenum)
            for j in range(featurenum):
                D[j]=torch.sum(I==j)
            stt=[]
            for j in range(featurenum):
                if D[j]>0:
                    K=names[j].copy()
                    K.append(int(D[j].item()))
                    stt.append(K)
            self.names.append(stt)
    def forward(self,x,w):
        bnum=x.shape[0]
        ysp=x.shape[2]
        xsp=x.shape[3]
        spread=self.spread
        featurenum=self.featurenum
        initwidth=self.initwidth
        receptive_field=self.receptive_field
        y=torch.zeros(bnum,w.shape[1],ysp-2*spread,xsp-2*spread)
        for j in range(featurenum):
            j_=j
            j1=j_%receptive_field
            j_=j_//receptive_field
            j2=j_%receptive_field
            j_=j_//receptive_field
            j3=j_%initwidth

            y0=spread-(j1-spread)
            y1=ysp-spread-(j1-spread)

            x0=spread-(j2-spread)
            x1=xsp-spread-(j2-spread)
            for i in range(w.shape[1]):
                y[:,i,:,:]+=w[j3,i]*x[:,j3,y0:y1,x0:x1]
        return y
    def cross_products(self, x,y,mask=[]):
        bnum=x.shape[0]
        ysp=x.shape[2]
        xsp=x.shape[3]
        spread=self.spread
        outwidth=self.outwidth
        initwidth=self.initwidth
        receptive_field=self.receptive_field
        x_=torch.zeros(bnum,outwidth+1,ysp-2*spread,xsp-2*spread)
        for j in range(outwidth):
            j_=j
            j1=j_%receptive_field
            j_=j_//receptive_field
            j2=j_%receptive_field
            j_=j_//receptive_field
            j3=j_%initwidth

            y0=spread-(j1-spread)
            y1=ysp-spread-(j1-spread)

            x0=spread-(j2-spread)
            x1=xsp-spread-(j2-spread)

            x_[:,j,:,:]=x[:,j3,y0:y1,x0:x1]
        x_[:,outwidth]=x_[:,outwidth]+1.
        #x=self.nn_layers[0](x)
        x=x_
        bnum=x.shape[0]
        nchan=x.shape[1]
        outnchan=y.shape[1]
        ysp=x.shape[2]
        xsp=x.shape[3]
        sp=ysp*xsp
        if len(mask)>0:
            x=x*mask
            y=y*mask
        x=torch.reshape(x,(bnum,nchan,sp))
        y=torch.reshape(y,(bnum,outnchan,sp))
        x=x.permute((1,0,2))
        y=y.permute((1,0,2))
        x=torch.reshape(x,(nchan,bnum*sp))
        y=torch.reshape(y,(outnchan,bnum*sp))
        degree=self.degree
        nfeat=len(self.res)
        x=x.to(torch.device("cpu"))
        y=y.to(torch.device("cpu"))
        X=torch.zeros(nfeat,bnum*sp).to(torch.device("cpu"))
        for i in range(nfeat):
            X[i]=torch.prod(x[self.res[i]],dim=0)
        X2=X@X.T
        XY=X@y.T
        Y2=torch.sum(torch.square(y),dim=1)
        return X2,XY,Y2
