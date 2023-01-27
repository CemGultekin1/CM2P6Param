from argparse import Namespace
from models.index import get_dict
from models.variations import lcnn_architecture
import numpy as np
from models.nets.cnn import CNN
from data.load import physical_domains
from models.index import update_model_info
from models.variations import qcnn_architecture,unet_architecture
from models.nets.others import QCNN,UNET,GAN
from models.regression import RegressionModel
from utils.parallel import get_device


def chan_nums(modelargs):
    ninchans = 2
    noutchans = 4
    if modelargs.temperature:
        ninchans += 1
        noutchans +=2
    if modelargs.latitude:
        ninchans += 2
    return ninchans,noutchans
def init_architecture(archargs:Namespace)->CNN:
    net=CNN(**archargs.__dict__)
    net = net.to(get_device())
    return net



def golden_model_bank(args,descriptive=False,configure=False,verbose=True,only_description=False):
    model_id=int(args.model_id)
    model_bank_id=args.model_bank_id

    if not only_description:
        data_info=get_dict(model_bank_id,model_id)

    folder_root='/scratch/zanna/data/cm2.6/'
    data_root=['coarse-surf-data-sigma-',\
                    'coarse-3D-data-sigma-',\
                    'coarse-1pct-CO2-surf-data-sigma-',\
                    'coarse-1pct-CO2-3D-data-sigma-']
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
        print(sigma_id,sigma_vals)
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
    args.data_address=folder_root+data_root[surf_deep]+str(sigma)


    args.data_address+='.zarr'

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
    elif arch_id==1: #QCNN
        widths,filter_size__,qwidth,qfilt=qcnn_architecture(sigma)
        net=QCNN(width=widths,qwidth=qwidth,filter_size=filter_size__,qfilt=qfilt,\
                 initwidth=inwidth,outwidth=outwidth,\
                 nprecision=outwidth,\
                 latsig=lat_features,\
                 latsign=lat_features,\
                 longitude=long_features,\
                 freq_coord=lat_features)
    elif arch_id==2: #UNET
        widths,filter_size__,deep_filters=unet_architecture(sigma)
        net=UNET(widths=widths,deep_filters=deep_filters,filter_size=filter_size__,\
                 initwidth=inwidth,outwidth=outwidth,\
                 nprecision=outwidth,\
                 latsig=lat_features,\
                 latsign=lat_features,\
                 longitude=long_features,\
                 freq_coord=lat_features)
    elif arch_id==3: #GAN
        net=GAN(initwidth=inwidth,outwidth=outwidth,\
                latsig=lat_features,\
                 latsign=lat_features,\
                 longitude=long_features,\
                 freq_coord=lat_features)
    elif arch_id==4: #Regression
        net=RegressionModel(initwidth=inwidth,\
                 latsig=lat_features,\
                 latsign=lat_features,\
                 direct_coord=direct_coords)
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
    if arch_id!=3:
        loss=lambda output, target, mask: \
            lossfun(output, target, mask,heteroscedastic=True,outsize=outwidth)
    else:
        loss=lambda output, target, mask: \
            ganlossfun(output, target, mask)
    description=model_names[arch_id]
    if arch_id==0:
        stt=str(filter_size)
        stt=stt+'x'+stt
        description+=' + '+stt
    if surf_deep%2==0:
        description+=' + '+'surface'
    elif surf_deep%2==1:
        depthval=depthvals[depthind]
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

    description+=' + '+'coarse('+str(sigma)+')'
    if verbose:
        print(description+' + '+'batch= '+str(args.batch), flush=True)
    partition=physical_domains(physical_dom_id)
    ds_zarr=load_ds_zarr(args)
    model_bank_id='G'

    if configure:
        data_info['direct_coord']=direct_coords
        data_info['freq_coord']=lat_features
        data_info['lat_feat']=lat_features
        data_info['long_feat']=long_features
        data_info['inputs']="usurf vsurf surface_temp".split()
        if not temperature[0]:
            data_info['inputs']=data_info['inputs'][:2]

        if residue_training:
            data_info['outputs']="Su_r Sv_r ST_r".split()
        else:
            data_info['outputs']="Su Sv ST".split()
        if not temperature[1]:
            data_info['outputs']=data_info['outputs'][:2]
        maskloc='/scratch/cg3306/climate/masks/'
        if surf_deep==0:
            maskloc+='surf'
        elif surf_deep==1:
            maskloc+='deep'
        maskloc+='-sigma'+str(sigma)
        maskloc+='-filter'+str(filter_size)
        if physical_dom_id==0:
            maskloc+='-dom4'
        if physical_dom_id==3:
            maskloc+='-glbl'
        if resnet:
            maskloc+='-padded'
        maskloc+='.npy'
        data_info['maskloc']=maskloc
    if not descriptive and configure:
        update_model_info(data_info,model_bank_id,model_id)
    # data_init=lambda partit : Dataset2(ds_zarr,partit,model_id,model_bank_id,\
    #                                                 net,subtime=args.subtime,parallel=args.nworkers>1,\
    #                                                 depthind=depthind)

    if not descriptive:
        return net,loss,data_init,partition
    else:
        return description
