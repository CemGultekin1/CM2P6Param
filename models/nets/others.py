import numpy as np
from models.nets.base import ClimateNet
import torch
from torch import nn
from torch.nn import functional as F
import copy




class QCNN(ClimateNet):
    def __init__(self,qwidth=64,qfilt=[11,11],spread=0,heteroscedastic=True,coarsen=0,                    width=[128,64,32,32,32,32,32,1],                    filter_size=[5,5,3,3,3,3,3,3],                    latsig=False,                    latsign=False,                    freq_coord=False,                    direct_coord=False,                    timeshuffle=False,                    physical_force_features=False,                    longitude=False,                    rescale=[1/10,1/1e7],                    initwidth=2,                    outwidth=2,                    nprecision=1):
        super(QCNN, self).__init__(spread=spread,coarsen=coarsen,latsig=latsig,timeshuffle=timeshuffle)
        device=self.device
        self.nn_layers = nn.ModuleList()
        spread=0

        self.rescale=torch.tensor(rescale,dtype=torch.float32,requires_grad=False)
        self.freq_coord=freq_coord
        self.heteroscedastic=heteroscedastic
        self.initwidth=initwidth
        self.outwidth=outwidth
        self.width=width
        self.filter_size=filter_size
        self.num_layers=len(filter_size)
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsign=latsign
        self.latsig=latsig
        self.nprecision=nprecision
        self.physical_force_features=physical_force_features
        self.nparam=0
        self.bnflag=True

        self.nn_layers.append(nn.Conv2d(initwidth, qwidth, qfilt[0]).to(device) )
        self.nparam+=initwidth*qwidth*qfilt[0]**2
        self.nn_layers.append(nn.BatchNorm2d(qwidth).to(device) )
        self.nparam+=qwidth
        self.nn_layers.append(nn.BatchNorm2d(qwidth).to(device) )
        self.nparam+=qwidth
        self.nn_layers.append(nn.Conv2d(qwidth, outwidth, qfilt[1]).to(device) )
        self.nparam+=outwidth*qwidth*qfilt[1]**2
        self.nn_layers.append(nn.Conv2d(initwidth, width[0], filter_size[0]).to(device) )
        self.nparam+=initwidth*width[0]*filter_size[0]**2
        spread+=(filter_size[0]-1)/2
        width[-1]=nprecision
        for i in range(1,self.num_layers):
            self.nn_layers.append(nn.BatchNorm2d(width[i-1]).to(device) )
            self.nparam+=width[i-1]
            self.nn_layers.append(nn.Conv2d(width[i-1], width[i], filter_size[i]).to(device) )
            self.nparam+=width[i-1]*width[i]*filter_size[i]**2
            spread+=(filter_size[i]-1)/2

        self.nn_layers.append(nn.Softplus().to(device))
        spread+=coarsen
        self.spread=np.int64(spread)
        self.receptive_field=self.spread*2+1
    def forward(self, x):
        u=x*1
        cn=0
        u = self.nn_layers[cn](u)
        cn+=1
        u = torch.square(self.nn_layers[cn](u))
        cn+=1
        u = self.nn_layers[cn](u)
        cn+=1
        u = self.nn_layers[cn](u)
        cn+=1

        for i in range(self.num_layers-1):
            x = self.nn_layers[cn](x)
            cn+=1
            x = F.relu(self.nn_layers[cn](x))
            cn+=1

        x=self.nn_layers[cn](x)
        cn+=1
        precision=self.nn_layers[cn](x)
        x=torch.cat([u,precision],dim=1)
        return x







class UNET(ClimateNet):
    def __init__(self,spread=0,heteroscedastic=True,                    #width=[128,64,32,32,32,32,32,3],\
                     widths=[64,128,256,512],\
                    pools=[2,2,2],\
                    filter_size=[5,5,3,3,3,3,3,3],\
                    deep_filters=[[3,3,3,1,1],[3,3,3,1,1],[3,3,3,1,1]],\
                    latsig=False,\
                    latsign=False,\
                    direct_coord=False,\
                    freq_coord=False,\
                    timeshuffle=False,\
                    physical_force_features=False,\
                    longitude=False,\
                    rescale=[1/10,1/1e7],\
                    initwidth=2,\
                    outwidth=2,\
                    nprecision=1,\
                    verbose=False):
        super(UNET, self).__init__()
        device=self.device
        bnflag=True
        self.bnflag=bnflag
        self.direct_coord=direct_coord
        self.freq_coord=freq_coord
        self.longitude=longitude
        self.latsig=latsig
        self.latsign=latsign
        self.nn_layers=nn.ModuleList()
        self.verbose=verbose
        self.nprecision=nprecision
        self.nparam=0
        self.bnflag=True#self.latsig or self.latsign or self.direct_coord
        if self.direct_coord:
            initwidth+=3
        elif self.freq_coord:
            if self.latsig:
                initwidth+=1
            if self.latsign:
                initwidth+=1
            if self.longitude:
                initwidth+=2
        self.initwidth=initwidth
        self.outwidth=outwidth
        self.spread=int((np.sum(filter_size)-len(filter_size))/2)



        widthin=initwidth
        widthout=outwidth+nprecision

        #widths=[64,128,256,512]#[2,4,8,16]#
        nlevel=len(widths)
        self.nlevel=nlevel
        self.locs=[]
        self.pools=pools

        self.receptive_field=1
        for i in range(len(pools)):
            ww=np.sum(deep_filters[-1-i])-len(deep_filters[-1-i])
            self.receptive_field=(self.receptive_field+ww)*pools[-i-1]
        self.receptive_field+=np.sum(filter_size[:3])-3
        #self.add_conv_layers([5,5,3],[widths[0]]*4,widthin=widthin)
        self.add_conv_layers(filter_size[:3],[widths[0]]*4,widthin=widthin)
        self.add_conv_layers(1,[widths[0]]*3)
        for i in range(nlevel-1):
            pool=[pools[i],pools[i]]
            self.add_down_sampling(pool)
            self.add_conv_layers(deep_filters[i],[widths[i+1]]*(len(deep_filters[i])+1),widthin=widths[i])
            #self.add_conv_layers(1,[widths[i+1]]*3)


        for i in range(nlevel-2):
            pool=[pools[-1-i],pools[-1-i]]
            self.add_up_sampling(1,[widths[-1-i],widths[-1-i]//2],pool)
            self.add_conv_layers(deep_filters[-1-i],[widths[-2-i]]*(len(deep_filters[-1-i])+1),widthout=widths[-2-i],widthin=widths[-1-i])


        i=nlevel-2
        pool=[pools[-1-i],pools[-1-i]]
        self.add_up_sampling(1,[widths[-1-i],widths[-1-i]//2],pool)
        self.add_conv_layers(filter_size[3:],[widths[-2-i]]*6,widthout=widthout,widthin=widths[-1-i])
        self.to_device()
        self.precisionlyr=nn.Softplus().to(device)
    def add_conv_layers(self,conv,width,widthin=0,widthout=0,final_nonlinearity=False):
        loc0=len(self.nn_layers)
        if widthin!=0:
            width[0]=widthin
        if widthout!=0:
            width[-1]=widthout
        if type(conv)==int:
            conv=[conv]*(len(width)-1)
        for i in range(len(width)-2):
            self.nn_layers.append(nn.Conv2d(width[i], width[i+1],conv[i]))
            self.nparam+=width[i]*width[i+1]*conv[i]**2
            self.nn_layers.append(nn.BatchNorm2d(width[i+1]))
            self.nparam+=width[i+1]
            self.nn_layers.append(nn.ReLU(inplace=True))
        self.nn_layers.append(nn.Conv2d(width[-2], width[-1],conv[-1]))
        self.nparam+=width[-2]*width[-1]*conv[-1]**2
        if final_nonlinearity:
            self.nn_layers.append(nn.BatchNorm2d(width[-1]))
            self.nparam+=width[-1]
            self.nn_layers.append(nn.ReLU(inplace=True))
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
    def add_down_sampling(self,pool):
        loc0=len(self.nn_layers)
        self.nn_layers.append(nn.MaxPool2d(pool, stride=pool))
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
    def add_up_sampling(self,conv,width,pool):
        loc0=len(self.nn_layers)
        self.nn_layers.append(nn.ConvTranspose2d(width[0], width[1],conv,stride=pool))
        self.nparam+=width[0]*width[1]*conv**2
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
    def apply_layers(self,x,K):
        for i in range(self.locs[K][0],self.locs[K][1]):
            x=self.nn_layers[i](x)
        return x
    def to_device(self,):
        for i in range(len(self.nn_layers)):
            self.nn_layers[i]= self.nn_layers[i].to(self.device)

    def trim_merge(self,f,x):
        if type(x)==torch.Tensor:
            ny=(f.shape[2]-x.shape[2])
            nx=(f.shape[3]-x.shape[3])
        else:
            ny=(f.shape[2]-x[0])
            nx=(f.shape[3]-x[1])
        ny0,ny1=ny//2,ny//2
        nx0,nx1=nx//2,nx//2
        if ny0+ny1<ny:
            ny1+=1
        if nx0+nx1<nx:
            nx1+=1
        f=f[:,:,ny0:-ny1,nx0:-nx1]

        if type(x)==torch.Tensor:
            return torch.cat([f,x],dim=1)
        else:
            return f
    def zeropad(self,f,x):
        if type(x)==torch.Tensor:
            diffY=(f.shape[2]-x.shape[2])
            diffX=(f.shape[3]-x.shape[3])
        else:
            diffY=(f.shape[2]-x.shape[2])
            diffX=(f.shape[3]-x.shape[3])

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if type(x)==torch.Tensor:
            return torch.cat([f,x],dim=1)
        else:
            return f
    def forward(self,u):
        locs=self.locs

        features=[]
        nlevel=self.nlevel
        t=0
        x=u*1
        x=self.apply_layers(x,t) # convolutions
        if self.verbose:
            print('conv: '+str(0)+' '+str(t)+  '  '+ str(x.shape))
        t+=1

        f=x*1
        f=self.apply_layers(f,t) # ptswise
        if self.verbose:
            print('ptswise: '+str(0)+' '+str(t)+  '  '+ str(f.shape))
        t+=1
        features.append(f)

        for i in range(nlevel-1):
            x=self.apply_layers(x,t) # downsampling
            if self.verbose:
                print('down: '+str(i+1)+' '+str(t)+  '  '+ str(x.shape))
            t+=1

            x=self.apply_layers(x,t) # convolutions
            if self.verbose:
                print('conv: '+str(i+1)+' '+str(t)+  '  '+ str(x.shape))
            t+=1

            f=x*1
            '''
            f=self.apply_layers(f,t) # ptswise
            if self.verbose:
                print('ptswise: '+str(i+1)+' '+str(t)+  '  '+ str(f.shape))
            t+=1'''
            features.append(f)

        f=features[-1]
        for jj in range(1,nlevel):
            j=nlevel-jj-1
            x=self.apply_layers(f,t) # upsample
            if self.verbose:
                print('upsample: ('+str(j)+ ', '+str(0)+') '+str(t)+ '  '+ str(x.shape))
            t+=1
            f=features[j]
            f=self.zeropad(f,x)
            f=self.apply_layers(f,t) # convolutions
            if self.verbose:
                print('conv: '+str(i+1)+' '+str(t)+  '  '+ str(f.shape))
            t+=1
        (mean,prec)=torch.split(f,[self.outwidth,self.nprecision],dim=1)
        prec=self.precisionlyr(prec)
        y=torch.cat([mean,prec],dim=1)
        return y






class GAN(ClimateNet):
    def __init__(self,spread=0,                    width_generator=[3,128,64,32,32,32,32,32,3],                    filter_size_generator=[3,3,3,3,3,3,3,3],                    width_discriminator=[3,128,64,32,32,32,32,32,1],                    filter_size_discriminator=[9,9,3,3,1,1,1,1],                    latsig=False,                    latsign=False,                    direct_coord=False,                    freq_coord=False,                    longitude=False,                    initwidth=3,                    outwidth=3,                    random_field=1):
        super(GAN, self).__init__(gan=True)
        device=self.device
        self.freq_coord=freq_coord

        self.outwidth=outwidth
        self.initwidth=initwidth
        self.random_field=random_field
        self.width_generator=width_generator
        self.width_discriminator=width_discriminator
        self.filter_size_generator=filter_size_generator
        self.filter_size_discriminator=filter_size_discriminator
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsign=latsign
        self.latsig=latsig
        self.generator_layers=[]
        self.discriminator_layers=[]

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

        # Discriminator build
        discriminator=ClimateNet()
        width=copy.deepcopy(width_discriminator)
        filter_size=copy.deepcopy(filter_size_discriminator)
        spread=0
        self.nparam=0
        width[0]=initwidth
        for i in range(len(filter_size)):
            if i==2:
                discriminator.nn_layers.append(nn.BatchNorm2d(outwidth).to(device) )
                self.nparam+=outwidth
                width[i]+=outwidth
            discriminator.nn_layers.append(nn.Conv2d(width[i], width[i+1], filter_size[i]).to(device) )
            self.nparam+=width[i]*width[i+1]*filter_size[i]**2
            spread+=(filter_size[i]-1)/2

            discriminator.nn_layers.append(nn.BatchNorm2d(width[i+1]).to(device) )
            self.nparam+=width[i+1]
            if i<len(filter_size)-1:
                discriminator.nn_layers.append(nn.ReLU(inplace=True).to(device))
        self.receptive_field=np.int64(spread*2+1)
        self.discriminator=discriminator

        # Generator build
        generator=ClimateNet()
        width=copy.deepcopy(width_generator)
        filter_size=copy.deepcopy(filter_size_generator)
        spread=0
        width[0]=initwidth+random_field
        width[-1]=outwidth
        for i in range(len(filter_size)):
            generator.nn_layers.append(nn.Conv2d(width[i], width[i+1], filter_size[i]).to(device) )
            self.nparam+=width[i]*width[i+1]*filter_size[i]**2
            spread+=(filter_size[i]-1)/2
            if i<len(filter_size)-1:
                generator.nn_layers.append(nn.BatchNorm2d(width[i+1]).to(device) )
                self.nparam+=width[i+1]
                generator.nn_layers.append(nn.ReLU(inplace=True).to(device))
            else:
                generator.nn_layers.append(nn.BatchNorm2d(width[i+1]).to(device) )
                self.nparam+=width[i+1]
        self.spread=np.maximum(np.int64(spread),self.spread)
        self.generator=generator
    def discriminator_forward(self,x,y):#,yhat):
        for i in range(6):
            x = self.discriminator.nn_layers[i](x)
        i=6
        #y=torch.cat([y,yhat],dim=1)
        y =self.discriminator.nn_layers[i](y)#-yhat)
        x=torch.cat([x,y],dim=1)
        for i in range(7,len(self.discriminator.nn_layers)):
            x = self.discriminator.nn_layers[i](x)
        return 1/(1+torch.exp(x))
    def generator_forward(self,x,z):
        x=torch.cat([x,z],dim=1)
        for i in range(len(self.generator.nn_layers)):
            x = self.generator.nn_layers[i](x)
        return x#torch.tanh(x)*50
