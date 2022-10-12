import numpy as np
from models.nets.others import QCNN,UNET



def approximate_widths(def_width,def_filters,filters):
    nlyr=len(def_filters)
    num_param=np.zeros(nlyr)
    for i in range(nlyr):
        num_param[i]=def_width[i]*def_width[i+1]*(def_filters[i]**2)/(filters[i]**2)
    widths=np.zeros(nlyr+1)
    widths[0]=def_width[0]
    for i in range(1,nlyr+1):
        widths[i]=int(num_param[i-1]/widths[i-1])
    widths[-1]=def_width[-1]
    widths=[int(w) for w in widths]
    return widths[1:]
def lcnn_architecture(ninchans,noutchans,width_scale,nlayers,filter_size,mode=0):
    widths=[ninchans,128,64,32,32,32,32,32,noutchans]
    widths=[int(np.ceil(width_scale*w)) for w in widths]
    widths = widths[:nlayers+1]
    widths[0],widths[-1] = ninchans,noutchans

    filters21=[5,5,3,3,3,3,3,3]

    cursize=21
    filters=np.array(filters21)
    while cursize>filter_size:
        filters=filter_shrink_method(filters,mode)
        cursize=np.sum(filters)-len(filters)+1

    filters = filters[:nlayers]
    assert cursize == filter_size
    
    return widths,filters
    
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




def unet_receptive_field_compute(filter_size,pools,deep_filters):
    receptive_field=1
    for i in range(len(pools)):
        ww=np.sum(deep_filters[-1-i])-len(deep_filters[-1-i])
        receptive_field=(receptive_field+ww)*pools[-i-1]
    receptive_field+=np.sum(filter_size[:3])-3
    return receptive_field
def unet_architecture(sigma):
    sigma1=4
    receptive_field1=102
    receptive_field=(int(receptive_field1/sigma*sigma1)//2+1)*2
    filter_size=[5,5,3,3,3,3,3,3]
    deep_filters=[[3,3,3,1,1,1],[3,3,3,1,1,1],[3,3,3,1,1,1]]
    widths=[64,128,256,512]
    if sigma==sigma1:
        return widths,filter_size,deep_filters
    pools=[2,2,2]
    org_filter_size__=copy.deepcopy(filter_size)
    org_deep_filters__=copy.deepcopy(deep_filters)
    rec=unet_receptive_field_compute(filter_size,pools,deep_filters)
    nomoreleft=False
    while rec>receptive_field:
        filter_size__=copy.deepcopy(filter_size)
        deep_filters__=copy.deepcopy(deep_filters)
        lvl=len(deep_filters)-1
        while lvl>=0:
            dlvl=np.array(deep_filters[lvl])
            if not np.all(dlvl==1):
                break
            lvl-=1
        if lvl>=0:
            I=np.where(dlvl>1)[0][-1]
            deep_filters[lvl][I]-=1
        else:
            ff=np.array(filter_size[:3])
            if not np.all(ff==1):
                I=np.where(ff>1)[0][-1]
                filter_size[I]-=2
            else:
                nomoreleft=True
                break
        rec=unet_receptive_field_compute(filter_size,pools,deep_filters)
    if nomoreleft:
        nomoreleft=False
        while rec>receptive_field:
            filter_size__=copy.deepcopy(filter_size)
            deep_filters__=copy.deepcopy(deep_filters)
            ff=np.array(filter_size[3:])
            if not np.all(ff==np.amax(ff)):
                I=np.where(ff==np.amax(ff))[0][-1]
                filter_size[I+3]-=2
            elif not np.all(ff==1):
                I=np.where(ff>1)[0][-1]
                filter_size[I+3]-=2
            else:
                nomoreleft=True
                break
            rec=unet_receptive_field_compute(filter_size,pools,deep_filters)
        if nomoreleft:
            print('WTF!')
        else:
            filter_size=copy.deepcopy(filter_size__)
            deep_filters=copy.deepcopy(deep_filters__)
    else:
        filter_size=copy.deepcopy(filter_size__)
        deep_filters=copy.deepcopy(deep_filters__)
    rec=unet_receptive_field_compute(filter_size,pools,deep_filters)

    net=UNET()
    nparam0=net.nparam
    net=UNET(filter_size=filter_size,deep_filters=deep_filters)
    nparam1=net.nparam
    rat=np.sqrt(nparam0/nparam1)
    
    widths=[int(np.ceil(w*rat)) for w in widths]
    net=UNET(widths=widths,filter_size=filter_size,deep_filters=deep_filters)
    nparam2=net.nparam
    return widths,filter_size,deep_filters

def qcnn_receptive_field_compute(filter_size):
    return np.sum(filter_size)-len(filter_size)+1
def qcnn_architecture(sigma):
    sigma1=4
    
    receptive_field1=21
    receptive_field=int(receptive_field1/sigma*sigma1)//2*2+1
    
    filter_size=[5,5,3,3,3,3,3,3]
    qwidth=64
    widths=[128,64,32,32,32,32,32,1]
    if sigma==sigma1:
        return widths,filter_size,qwidth,[11,11]
    
    qfilt1=(receptive_field+1)//2
    if qfilt1%2==0:
        qfilt2=qfilt1-1
        qfilt1+=1
    else:
        qfilt2=qfilt1
    qfilt=[qfilt1,qfilt2]
    
    org_filter_size__=copy.deepcopy(filter_size)
    rec=qcnn_receptive_field_compute(filter_size)
    nomoreleft=False
    
    while rec>receptive_field:
        filter_size__=copy.deepcopy(filter_size)
        ff=np.array(filter_size)
        if not np.all(ff==np.amax(ff)):
            I=np.where(ff==np.amax(ff))[0][-1]
            filter_size[I]-=2
        elif not np.all(ff==1):
            I=np.where(ff>1)[0][-1]
            filter_size[I]-=2
        else:
            nomoreleft=True
            break
        rec=qcnn_receptive_field_compute(filter_size)
    if nomoreleft:
        print('WTF!')
    net=QCNN()
    nparam0=net.nparam
    net=QCNN(filter_size=filter_size,qfilt=qfilt)
    nparam1=net.nparam
    rat=np.sqrt(nparam0/nparam1)
    
   
    
    qwidth=int(np.ceil(qwidth*rat**2))
    widths=[int(np.ceil(w*rat)) for w in widths]

    net=QCNN(width=widths,qwidth=qwidth,filter_size=filter_size,qfilt=qfilt)
    nparam2=net.nparam
    return widths,filter_size,qwidth,qfilt

