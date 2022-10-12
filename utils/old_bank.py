import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from argparse import Namespace

def golden_model_bank(args,verbose=True,only_description=False):
    model_id=int(args.model_id)
    # folder_root='/scratch/zanna/data/cm2.6/'
    # data_root=['coarse-surf-data-sigma-',\
    #                 'coarse-3D-data-sigma-',\
    #                 'coarse-1pct-CO2-surf-data-sigma-',\
    #                 'coarse-1pct-CO2-3D-data-sigma-']
    newargs = Namespace()
    model_names=['LCNN','QCNN','UNET','GAN','REG','LCNN-SIZE-PLAY']
    sigma_vals=[4,8,12,16]
    filter_sizes=[21,15,9,7,5,3,1]

    STEP=1000
    test_type=model_id//STEP
    test_num=model_id%STEP
    # default parameter choices
    surf_deep=0
    lat_features=False
    long_features=False
    direct_coords=False
    residue_training=False
    temperature=[True,True]
    co2test_flag= args.co2==1
    physical_dom_id=0
    depthvals=[5.03355 , 55.853249,  110.096153, 181.312454,  330.007751,1497.56189 , 3508.633057]
    sigma_id=0
    filt_mode=0
    arch_id=0
    filter_size=21
    outwidth=3
    inwidth=3
    depthind=2
    resnet=False
    # index parameter init
    tt=test_num
    if test_type==1:
        C=[2,7,2,4,2]
        title='depth test'
        names=[['surf/depth','surf','depth'],\
                    ['sizes','21-default','5-default','5-8layer-thin',\
                         '5-8layer-thinner','5-6layer','5-4layer','5-3layer'],\
                        ['res','no','yes'],\
                              ['sigma']+[str(i) for i in sigma_vals],\
                                  ['global','no','yes']]


        surf_deep=tt%2
        tt=tt//2
        if surf_deep:
            depthind=2
        temperature[0]=0
        temperature[1]=0

        if not temperature[0]:
            inwidth=2
        if not temperature[1]:
            outwidth=2

        arch_id=5

        imodel=tt%7
        tt=tt//7

        if imodel==0:
            widths=[128,64,32,32,32,32,32,3]
            filters=[5,5,3,3,3,3,3,3]
            filter_size=21
        elif imodel==1:
            widths,filters,_=lcnn_architecture(1.,5,mode=0)
            filter_size=5
        elif imodel==2:
            widths=[128,64,32,32,32,32,32,3]
            filters=[3,3,1,1,1,1,1,1]
            filter_size=5
        elif imodel==3:
            widths=[64,32,16,16,16,16,16,3]
            filters=[3,3,1,1,1,1,1,1]
            filter_size=5
        elif imodel==4:
            widths=[32,16,16,16,16,3]
            filters=[3,3,1,1,1,1]
            filter_size=5
        elif imodel==5:
            widths=[32,16,16,3]
            filters=[3,3,1,1]
            filter_size=5
        elif imodel==6:
            widths=[32,16,3]
            filters=[3,3,1]
            filter_size=5



        residue_training=(tt%2)!=0
        tt=tt//2

        physical_dom_id=3



        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)
        sigma=sigma_vals[sigma_id]

        phys_code=tt%2
        tt=tt//2

        if phys_code==0:
            physical_dom_id=0
        else:
            physical_dom_id=3

        if physical_dom_id==3:
            args.batch=int(2*(sigma/4)**2)
        else:
            args.batch=64
    elif test_type==2:
        # COARSE-GRAIN
        physical_dom_id=3
        args.batch=2
        surf_deep=tt%2
        tt=tt//2
        sigma_id=tt%(len(sigma_vals))
        sigma=sigma_vals[sigma_id]
        args.batch=int(2*(sigma/4)**2)
        filter_size=np.int(np.ceil(21/sigma*4))//2*2+1
    elif test_type==4:
        # FULL TYPE TRAINING

        # DATASET (2)
        # SURF/DEEP

        # FILTERSIZE (7)
        # 21 15 9 7 5 3 1

        # SIGMAVALS (4)
        # 4 8 12 16

        # GEOPHYS (3)
        # NONE - GLOBAL - (GLOBAL+COORDS)

        # RESIDUE TARGET(2)
        # YES - NO


        '''
        EXPERIMENT 1
            Filtersize + Sigmaval + GEOPHY
                15/9/5/1 + 4  + GLBCOORDS
                15/9/5/1 + 8  + GLBCOORDS
                15/9/5/1 + 16 + GLBCOORDS
            VALUES
               4114,4115,4116,4117,4120,4121,4124,4125,
               4128,4129,4130,4131,4134,4135,4138,4139,
               4156,4157,4158,4159,4162,4163,4166,4167
       EXPERIMENT 2
            Filtersize + Sigmaval + GEOPHY + NO RESIDUE
                15/9/5/1 + 4  + GLBCOORDS
                15/9/5/1 + 8  + GLBCOORDS
                15/9/5/1 + 16 + GLBCOORDS
            VALUES
               4282,4283,4284,4285,4288,4289,4292,4293,
               4296,4297,4298,4299,4302,4303,4306,4307,
               4324,4325,4326,4327,4330,4331,4334,4335
        '''

        surf_deep=tt%2
        tt=tt//2

        filter_size_id=tt%len(filter_sizes)
        tt=tt//len(filter_sizes)

        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)

        sigma=sigma_vals[sigma_id]
        filter_size=filter_sizes[filter_size_id]

        args.batch=256+64

        geophys=tt%3
        if geophys>0:
            physical_dom_id=3
            args.batch=int(2*(sigma/4)**2)
        if geophys==2:
            lat_features =True
            direct_coords=True

        tt=tt//3
        residue_training=(tt%2)==0
    elif test_type==5:
        # FULL TYPE TRAINING

        # DATASET (2)
        # SURF/DEEP

        # ARCHITECTURE (3)
        # LCNN/QCNN/UNET

        # SIGMAVALS (4)
        # 4 8 12 16

        # GEOPHYS (3)
        # NONE - GLOBAL - (GLOBAL+COORDS)

        # RESIDUE TARGET(2)
        # YES - NO
        '''
        EXPERIMENT 1
            Dataset (2) + LCNN/QCNN (2) + Sigmavals (4) + 4 Domains + Res
                5000,5001,5002,5003,5006,5007,5008,5009,5012,5013,5014,5015,5018,5019,5020,5021
        EXPERIMENT 2
            Dataset (2) + LCNN/QCNN/UNET (3) + Sigmavals (4) + GLBL + Res
            Dataset (2) + LCNN/UNET (2) + Sigmavals (4) + COORDS + Res
                5024,5025,5026,5027,5028,5029,5030,5031,5032,5033,5034,5035,5036,5037,5038,5039,\
                5040,5041,5042,5043,5044,5045,5046,5047,5048,5049,5052,5053,5054,5055,5058,5059,\
                5060,5061,5064,5065,5066,5067,5070,5071
        EXPERIMENT 1.5
            Dataset (2) + LCNN/QCNN (2) + Sigmavals (4) + 4 Domains + No Res
                5072,5073,5074,5075,5078,5079,5080,5081,5084,5085,5086,5087,5090,5091,5092,5093
        EXPERIMENT 2.5
            Dataset (2) + LCNN/QCNN/UNET (3) + Sigmavals (4) + GLBL + No Res
                5096,5097,5098,5099,5100,5101,5102,5103,5104,5105,5106,5107,5108,5109,5110,5111,5112,5113,5114,5115,5116,5117,5118,5119
        '''

        C=[2,3,4,3,2]
        title='full-training'
        names=[['dataset', 'surf','depth 110m'],\
                       ['architecture','LCNN','QCNN','UNET'],\
                       ['sigma']+[str(sig) for sig in sigma_vals],\
                       ['training-doms','4regions','global','global+coords'],\
                       ['residue','yes','no']]
        surf_deep=tt%2
        tt=tt//2

        arch_id=tt%3
        tt=tt//3

        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)

        sigma=sigma_vals[sigma_id]
        filter_size=int(21*4/sigma//2)*2+1
        args.batch=32*(sigma//4)#4#256

        geophys=tt%3
        if geophys>0:
            physical_dom_id=3
            args.batch=int(2*(sigma/4)**2)
        if geophys==2:
            lat_features =True
            #direct_coords=True
        tt=tt//3
        residue_training=(tt%2)==0
    elif test_type==6:
        '''
        Regression model with various settings
        EXPERIMENT
            Dataset (2) + Regression (1) + Sigmavals (4) + Training(3) + Res/No (2)
            6000-6048
        '''

        C=[2,4,3,2]
        title='linear regression'
        names=[['dataset', 'surf','depth 110m'],\
                       ['sigma']+[str(sig) for sig in sigma_vals],\
                       ['training-doms','4regions','global','global+coords'],\
                       ['residue','yes','no']]

        surf_deep=tt%2
        tt=tt//2

        arch_id=4

        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)

        sigma=sigma_vals[sigma_id]
        args.batch=4

        geophys=tt%3
        if geophys>0:
            args.batch=1
            physical_dom_id=3
        if geophys==2:
            lat_features =True
            direct_coords=True
        tt=tt//3
        residue_training=(tt%2)==0
    elif test_type==7:
        '''
        Testing various shrinkage types
        EXPERIMENT
            Sigmavals (4) + Shrinkage (6)
        '''

        C=[4,6]
        title='shrinkage procedures'
        names=[['sigma']+[str(sig) for sig in sigma_vals],\
                   ['shrinkage type']+[str(sig) for sig in range(6)]]

        sigma_id=tt%4
        tt=tt//4
        sigma=sigma_vals[sigma_id]
        filter_size=int(21*4/sigma//2)*2+1
        args.batch=4
        filt_mode=tt+1
    elif test_type==8:
        filter_sizes=[21,15,9,7,5,4,3,2,1]

        C=[2,9,4,2]
        title='filter size training'
        names=[['dataset', 'surf','depth 110m'],\
                       ['filter sozes']+[str(sig) for sig in filter_sizes],\
                       ['sigma']+[str(sig) for sig in sigma_vals],\
                       ['residue','yes','no']]
        filt_mode=1

        surf_deep=tt%2
        tt=tt//2

        filter_size_id=tt%len(filter_sizes)
        tt=tt//len(filter_sizes)

        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)

        residue_training=(tt%2)==0

        sigma=sigma_vals[sigma_id]
        filter_size=filter_sizes[filter_size_id]

        geophys=1
        physical_dom_id=3
        args.batch=int(2*(sigma/4)**2)
    elif test_type==9:
        C=[2,2,2,2,len(sigma_vals),3]
        title='root improvement'
        names=[['temp','no','yes'],\
                    ['global','no','yes'],\
                          ['res','no','yes'],\
                              ['geophys','no','yes'],\
                                ['sigma']+[str(sig) for sig in sigma_vals],\
                                      ['widths']+[str(sig) for sig in [0,1,2]]]


        resnet=True
        surf_deep=0
        temperature[0]=1-(tt%2 == 0)
        temperature[1]=1-(tt%2 == 0)
        tt=tt//2

        if not temperature[0]:
            inwidth=2
        if not temperature[1]:
            outwidth=2

        if tt%2==0:
            physical_dom_id=0
        else:
            physical_dom_id=3
        tt=tt//2


        residue_training=(tt%2)!=0
        tt=tt//2

        lat_features=(tt%2)!=0
        tt=tt//2

        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)
        # print(sigma_id,sigma_vals)
        sigma=sigma_vals[sigma_id]

        width_id=tt
        if sigma==4:
            # spread = 10
            filters=[3]*10+[1]*6
        elif sigma==8:
            # spread = 5
            filters=[3]*5+[1]*11
        elif sigma==12:
            # spread = 4
            filters=[3]*4+[1]*12
        elif sigma==16:
            # spread = 3
            filters=[3]*3+[1]*13
        widths=[[64,32,1],[128,64,1],[256,128,1]]
        widths=widths[width_id]

        if physical_dom_id==3:
            args.batch=int(2*(sigma/4)**2)
            if width_id==2:
                args.batch=int(args.batch/2)
        else:
            args.batch=165

        filter_size=int(21*4/sigma/2)*2+1
    elif test_type==3:
        C=[2,2,2,2,len(sigma_vals),2]
        title='root improvement'
        names=[['temp','no','yes'],\
                    ['global','no','yes'],\
                          ['res','no','yes'],\
                              ['geophys','no','yes'],\
                                  ['sigma']+[str(sig) for sig in sigma_vals],\
                                      ['depth']+['surface','110m']]


        surf_deep=0
        temperature[0]=1-(tt%2 == 0)
        temperature[1]=1-(tt%2 == 0)
        tt=tt//2
        if tt%2==0:
            physical_dom_id=0
        else:
            physical_dom_id=3
        tt=tt//2

        residue_training=(tt%2)!=0
        tt=tt//2

        lat_features=(tt%2)!=0
        #direct_coords=(tt%2)!=0
        tt=tt//2

        sigma_id=tt%len(sigma_vals)
        sigma=sigma_vals[sigma_id]
        tt=tt//len(sigma_vals)

        if physical_dom_id==3:
            args.batch=int(2*(sigma/4)**2)
        else:
            args.batch=120

        filter_size=int(21*4/sigma/2)*2+1
        if not temperature[0]:
            inwidth=2
        if not temperature[1]:
            outwidth=2

        surf_deep=tt%2
        tt=tt//2
        depthind=2
    elif test_type==0:
        C=[2,2,2,7,len(sigma_vals)]
        depthvals=[5.03355,55.853249,110.096153,181.312454,330.007751, 1497.56189 , 3508.633057]
        title='depth test'
        names=[['temp','yes','no'],\
                    ['res','no','yes'],\
                        ['geophys','no','yes'],\
                            ['training-depth']+[str(i) for i in range(7)],\
                              ['sigma']+[str(i) for i in sigma_vals]]


        surf_deep=1
        temperature[0]=1-(tt%2)
        temperature[1]=temperature[0]
        tt=tt//2

        if not temperature[0]:
            inwidth=2
        if not temperature[1]:
            outwidth=2


        residue_training=(tt%2)!=0
        tt=tt//2

        lat_features=(tt%2)!=0
        #direct_coords=(tt%2)!=0
        tt=tt//2

        physical_dom_id=3



        depthind=tt%7
        tt=tt//7


        sigma_id=tt
        sigma=sigma_vals[tt]

        if physical_dom_id==3:
            args.batch=int(2*(sigma/4)**2)
        else:
            args.batch=128
        filter_size=int(21*4/sigma/2)*2+1

    if only_description:
        title+=' '+str(STEP*test_type)
        if verbose:
            print(title)
            for i in range(len(names)):
                print('\t'+names[i][0])
                outputstr='\t\t'
                for j in range(1,len(names[i])):
                    outputstr+=names[i][j]+' - '
                print(outputstr)
        return C,names
    if co2test_flag:
        surf_deep+=2
    sigma=sigma_vals[sigma_id]

    if arch_id not in [0,5]:
        return None,None
    if arch_id==0: #LCNN
        width_scale=1
        if not resnet:
            widths,filters,nparam=lcnn_architecture(width_scale,filter_size,mode=filt_mode)
        else:
            _,filters,nparam=lcnn_architecture(width_scale,filter_size,mode=filt_mode)
        net=LCNN(initwidth=inwidth,outwidth=outwidth,\
                 filter_size=filters,\
                 width=widths,\
                 nprecision=outwidth,\
                 latsig=lat_features,\
                 latsign=lat_features,\
                 longitude=long_features,\
                 freq_coord=lat_features and not direct_coords,\
                 direct_coord=direct_coords,\
                 skipcons=resnet)
    elif arch_id==5: #LCNN-shrinked
        # widths and filters are already defined
        net=LCNN(initwidth=inwidth,outwidth=outwidth,\
                 filter_size=filters,\
                 width=widths,\
                 nprecision=outwidth,\
                 latsig=lat_features,\
                 latsign=lat_features,\
                 longitude=long_features,\
                 freq_coord=lat_features and not direct_coords,\
                 direct_coord=direct_coords)
    newargs.domain = 'global' if physical_dom_id==3 else 'four_regions'
    newargs.temperature = temperature[0]>0
    newargs.latitude = lat_features
    newargs.sigma =sigma
    newargs.linsupres = residue_training
    newargs.depth = depthvals[depthind] if surf_deep%2==1 else 0.
    newargs.seed = -1
    newargs.co2 = surf_deep//2 == 1
    newargs.normalization = 'absolute'


    initwidth = 2
    outwidth = 4
    if newargs.temperature:
        initwidth+=1
        outwidth+=2
    if newargs.latitude:
        initwidth+=2
    widths = [initwidth] + widths
    widths[-1] = outwidth

    newargs.kernels = filters
    newargs.widths= widths
    newargs.minibatch = args.batch
    newargs.skipconn = [int(resnet)]*len(newargs.kernels)
    newargs.batchnorm = [1]*(len(newargs.kernels)-1) + [0]



    description=model_names[arch_id]
    if arch_id==0:
        stt=str(filter_size)
        stt=stt+'x'+stt
        description+=' + '+stt
    if surf_deep%2==0:
        description+=' + '+'surface'
    elif surf_deep%2==1:
        depthval= newargs.depth
        depthval=str(int(np.round(depthval)))
        description+=' + '+'deep ('+str(depthval)+'m)'

    if surf_deep//2==1:
        description+=' +1%CO2'
    if residue_training:
        description+=' + '+'res'
    if physical_dom_id==0:
        description+=' + '+'4 domains'
    elif physical_dom_id==3:
        description+=' + '+'glbl'
        if lat_features:
            description+=' + '+'lat'
        if long_features:
            description+=' + '+'long'

    # data_init=lambda partit : climate_data.Dataset2(ds_zarr,partit,model_id,model_bank_id,\
    #                                                 net,subtime=args.subtime,parallel=args.nworkers>1,\
    #                                                 depthind=depthind)
    description+=' + '+'coarse('+str(sigma)+')'
    return newargs,description



def lcnn_architecture(width_scale,filter_size,mode=0):
    widths=[128,64,32,32,32,32,32,3]
    widths=[np.ceil(width_scale*w) for w in widths]
    filters21=[5,5,3,3,3,3,3,3]
    if filter_size<21:
        filter_size=(filter_size//2)*2+1
        cursize=21
        filters=np.array(filters21)
        while cursize>filter_size:
            filters=filter_shrink_method(filters,mode)
            cursize=np.sum(filters)-len(filters)+1
        filters = filters.tolist()
    else:
        filters=filters21
    #widths=approximate_widths(widths,filters21,filters)
    net=LCNN()
    nparam0=net.nparam
    net=LCNN(filter_size=filters)
    nparam1=net.nparam
    rat=np.sqrt(nparam0/nparam1)
    widths=[int(np.ceil(rat*w)) for w in widths]
    widths[-1]=3
    net=LCNN(filter_size=filters,width=widths)
    return widths,filters,net.nparam
def filter_shrink_method(filters,mode):
    if mode==0:
        # Default
        i=np.where(filters==np.amax(filters))[0][-1]
        filters[i]-=2
    elif mode==1:
        # top-to-bottom equal shrink
        i=np.where(filters==np.amax(filters))[0][-1]
        filters[i]-=1
    elif mode==2:
        # top-to-bottom aggressive shrink
        i=np.where(filters!=1)[0][-1]
        filters[i]-=1
    elif mode==3:
        # bottom-to-top aggressive shrink
        i=np.where(filters!=1)[0][0]
        filters[i]-=1
    else:
        np.random.seed(mode)
        order=np.argsort(np.random.rand(len(filters)))
        I=np.where(filters==np.amax(filters))[0]
        I=np.array([i for i in order if i in I])
        i=I[0]
        filters[i]-=1
    return filters




class ClimateNet(nn.Module):
    def __init__(self,spread=0,coarsen=0,rescale=[1/10,1/1e7],latsig=False,                 timeshuffle=True,direct_coord=True,longitude=False,latsign=False,gan=False):
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

def physical_forces(x):
    dudy=x[:,:2,2:,1:-1]-x[:,:2,:-2,1:-1]
    dudx=x[:,:2,1:-1,2:]-x[:,:2,1:-1,:-2]
    x=x[:,:,1:-1,1:-1]
    u_=x[:,0:1]
    v_=x[:,1:2]
    x=torch.cat([x,dudy,dudx,u_*dudy,v_*dudy,u_*dudx,v_*dudx],dim=1)
    return x

class LCNN(ClimateNet):
    def __init__(self,spread=0,heteroscedastic=True,coarsen=0,                    width=[128,64,32,32,32,32,32,3],                    filter_size=[5,5,3,3,3,3,3,3],                    latsig=False,                    latsign=False,                    direct_coord=False,                    freq_coord=False,                    timeshuffle=False,                    physical_force_features=False,                    longitude=False,                    rescale=[1/10,1/1e7],                    initwidth=2,                    outwidth=2,                    nprecision=1,                    skipcons=False):
        super(LCNN, self).__init__(spread=spread,coarsen=coarsen,latsig=latsig,timeshuffle=timeshuffle)
        device=self.device
        self.nn_layers = nn.ModuleList()
        spread=0
        self.nparam=0
        self.rescale=torch.tensor(rescale,dtype=torch.float32,requires_grad=False)
        self.freq_coord=freq_coord
        self.heteroscedastic=heteroscedastic
        width[-1]=outwidth+nprecision
        self.width=width
        self.outwidth=outwidth
        self.initwidth=initwidth
        self.filter_size=filter_size
        self.num_layers=len(filter_size)
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsign=latsign
        self.latsig=latsig
        self.nprecision=nprecision
        self.skipcons=skipcons
        self.physical_force_features=physical_force_features

        self.bnflag=True#self.latsig or self.latsign or self.direct_coord
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
        if physical_force_features:
            initwidth+=12
            i=len(filter_size)
            while i>0:
                i-=1
                if filter_size[i]>1:
                    break
            filter_size[i]-=2
            spread+=1

        self.padding=[ff//2*self.skipcons for ff in filter_size]
        if not self.skipcons:
            self.nn_layers.append(nn.Conv2d(initwidth, width[0], filter_size[0],padding=self.padding[0]).to(device) )

            self.nparam+=initwidth*width[0]*filter_size[0]**2
            spread+=(filter_size[0]-1)/2
            for i in range(1,self.num_layers):
                if self.bnflag:
                    self.nn_layers.append(nn.BatchNorm2d(width[i-1]).to(device) )
                    self.nparam+=width[i-1]
                self.nn_layers.append(nn.Conv2d(width[i-1], width[i], filter_size[i],padding=self.padding[i]).to(device) )
                self.nparam+=width[i-1]*width[i]*filter_size[i]**2
                spread+=(filter_size[i]-1)/2
        else:
            w0=width[0]
            w1=width[1]
            w2=width[-1]
            self.nn_layers.append(nn.Conv2d(initwidth, w0, 1,padding=0).to(device) )
            self.nn_layers.append(nn.BatchNorm2d(w0).to(device) )
            self.nparam+=initwidth*w0+w0
            for i in range(self.num_layers):
                self.nn_layers.append(nn.Conv2d(w0, w1, 1,padding=0).to(device) )
                self.nn_layers.append(nn.BatchNorm2d(w1).to(device) )
                self.nn_layers.append(nn.Conv2d(w1, w1, filter_size[i],padding=self.padding[i]).to(device) )
                self.nn_layers.append(nn.BatchNorm2d(w1).to(device) )
                self.nn_layers.append(nn.Conv2d(w1, w0, 1,padding=0).to(device) )
                self.nparam+=w0*w1*2+w1**2*filter_size[i]**2+w0+w1*2
                spread+=(filter_size[i]-1)/2
            self.nn_layers.append(nn.Conv2d(w0, w2, 1,padding=0).to(device) )
            self.nparam+=w2*w0
        self.nn_layers.append(nn.Softplus().to(device))
        spread+=coarsen
        self.spread=np.int64(spread)
        self.receptive_field=self.spread*2+1
    def forward(self, x):
        #x=x/self.rescale[0]
        #if self.physical_force_features:
        #    x=physical_forces(x)
        #x=self.coarse_grain(x,self.coarsen)
        if not self.skipcons:
            cn=0
            for i in range(self.num_layers-1):
                x = self.nn_layers[cn](x)
                cn+=1
                if self.bnflag:
                    x = F.relu(self.nn_layers[cn](x))
                    cn+=1
                else:
                    x = F.relu(x)
            x=self.nn_layers[cn](x)
            cn+=1
        else:
            cn=0
            x = self.nn_layers[cn](x)
            cn+=1
            x = self.nn_layers[cn](x)
            cn+=1
            for i in range(self.num_layers):
                init=x*1
                x = self.nn_layers[cn](x)
                cn+=1
                x = F.relu(self.nn_layers[cn](x))
                cn+=1
                x = self.nn_layers[cn](x)
                cn+=1
                x = F.relu(self.nn_layers[cn](x))
                cn+=1
                x = self.nn_layers[cn](x)
                cn+=1
                x+=init
            x = self.nn_layers[cn](x)
            cn+=1
        mean,precision=torch.split(x,[x.shape[1]-self.nprecision,self.nprecision],dim=1)
        precision=self.nn_layers[cn](precision)
        x=torch.cat([mean,precision],dim=1)
        return x
