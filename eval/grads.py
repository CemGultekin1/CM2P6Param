def grad_probe_features2(uv,uv1,g,yc,xc,listout=False,projection=[],geoflag=True):
    if listout:        
        '''[y_coords[mid],x_coords[mid],\
           uv5,nuv,duvdt,duvdy,duvdx,duvdyy,duvdxy,duvdxx,\
              g5,r,cl2]'''
        names=[]
        t=0
        names.append(['coords',0,t+2])
        t+=2
        names.append(['uv5',t,t+2*5**2])
        t+=2*5**2
        names.append(['nuv',t,t+2])
        t+=2
        names.append(['duvdt',t,t+2])
        t+=2
        names.append(['duvdy',t,t+2])
        t+=2
        names.append(['duvdx',t,t+2])
        t+=2
        names.append(['duvdyy',t,t+2])
        t+=2
        names.append(['duvdxy',t,t+2])
        t+=2
        names.append(['duvdxx',t,t+2])
        t+=2
        names.append(['g5',t,t+2*5**2])
        t+=2*5**2
        names.append(['r',t,t+2*11])
        t+=2*11
        if geoflag:
            names.append(['cl2',t,t+6])
            t+=6
        else:
            names.append(['cl2',t,t+4])
            t+=4
        if len(projection)>0:
            names.append(['l-g5',t,t+2*5**2])
            t+=2*5**2
            names.append(['l-r',t,t+2*11])
            t+=2*11
            if geoflag:
                names.append(['l-cl2',t,t+6])
                t+=6
            else:
                names.append(['l-cl2',t,t+4])
                t+=4
        return names
    
    width=len(xc)
    mid=(width-1)//2
    
    uv=torch.reshape(uv,[-1,width,width])
    uv1=torch.reshape(uv1,[-1,width,width])
    g=torch.reshape(g,[-1,width,width])
    nchan=uv.shape[0]
    

    
    uv,_=uv.split([2,uv.shape[0]-2],dim=0)
    uv1,_=uv1.split([2,uv1.shape[0]-2],dim=0)
    
    
    
    area=(yc[-1]-yc[0])*(xc[-1]-xc[0])
    
    dyc= yc[1:]-yc[:-1]
    dxc= xc[1:]-xc[:-1]
    
    
    ddyc= (dyc[1:] + dyc[:-1])/2
    ddxc= (dxc[1:] + dxc[:-1])/2
    
    dyc=dyc.reshape([1,-1,1])
    dxc=dxc.reshape([1,1,-1])
    ddyc=ddyc.reshape([1,-1,1])
    ddxc=ddxc.reshape([1,1,-1])
    
    duvdy=(uv[:,:-1]-uv[:,1:])/dyc
    duvdx=(uv[:,:,:-1]-uv[:,:,1:])/dxc
    
    duvdyy=(duvdy[:,:-1]-duvdy[:,1:])/ddyc
    duvdxy=(duvdy[:,:,:-1]-duvdy[:,:,1:])/dxc
    duvdxx=(duvdx[:,:,:-1]-duvdx[:,:,1:])/ddxc
    
    duvdy=torch.sum(duvdy**2,dim=[1,2])*area
    duvdx=torch.sum(duvdx**2,dim=[1,2])*area
    
    duvdyy=torch.sum(duvdyy**2,dim=[1,2])*area
    duvdxy=torch.sum(duvdxy**2,dim=[1,2])*area
    duvdxx=torch.sum(duvdxx**2,dim=[1,2])*area
    
    duvdt=torch.sum((uv1-uv)**2,dim=[1,2])*area
    

    nuv=(uv**2).sum(axis=(1,2))*area
    
    
   
    
    
    uv5=uv[:2,mid-2:mid+3,mid-2:mid+3]
    
    
    ng=(g**2).sum(axis=(1,2),keepdim=True)*area
    g5=g[:2,mid-2:mid+3,mid-2:mid+3]
    r=torch.zeros(2,mid+1)
    r[:,0]=g[:2,mid,mid]**2
    
    for i in range(1,mid+1):
        r[:,i]+=torch.sum(g[:2,mid-i:mid+i,mid+i]**2,dim=1)
        r[:,i]+=torch.sum(g[:2,mid+i,mid-i:mid+i]**2,dim=1)
        r[:,i]+=torch.sum(g[:2,mid-i:mid+i,mid-i]**2,dim=1)
        r[:,i]+=torch.sum(g[:2,mid-i,mid-i:mid+i]**2,dim=1)
        r[:,i]=r[:,i]/( (2*i+1)*4-4)
    cl2=ng/torch.sqrt(torch.sum(ng**2))
    
    F=[yc[mid:mid+1],xc[mid:mid+1],\
           uv5,nuv,duvdt,duvdy,duvdx,duvdyy,duvdxy,duvdxx,\
              g5,r,cl2]
    
    
    if len(projection)>0:
        g_=(projection@(projection.T@(g[:2].reshape([-1])))).reshape([-1,width,width])
        g=torch.cat([g_,g[2:]],axis=0)
        ng=(g**2).sum(axis=(1,2),keepdim=True)*area
        g5_=g[:2,mid-2:mid+3,mid-2:mid+3]
        r_=torch.zeros(2,mid+1)
        r_[:,0]=g[:2,mid,mid]**2
        for i in range(1,mid+1):
            r_[:,i]+=torch.sum(g[:2,mid-i:mid+i,mid+i]**2,dim=1)
            r_[:,i]+=torch.sum(g[:2,mid+i,mid-i:mid+i]**2,dim=1)
            r_[:,i]+=torch.sum(g[:2,mid-i:mid+i,mid-i]**2,dim=1)
            r_[:,i]+=torch.sum(g[:2,mid-i,mid-i:mid+i]**2,dim=1)
            r_[:,i]=r[:,i]/( (2*i+1)*4-4)
        cl2_=ng/torch.sqrt(torch.sum(ng**2))
        F=F+[g5_,r_,cl2_]
    F=[f.reshape([-1]) for f in F]
    return torch.cat(F,dim=0)
def grad_probe_features3(g,yc,xc,inchan,spread,listout=False,geoflag=False):
    if listout:        
        names=[]
        t=0
        names.append(['coords',0,t+2])
        t+=2
        names.append(['r',t,t+inchan*(spread+1)])
        t+=inchan*(spread+1)
        if geoflag:
            names.append(['cl2',t,t+inchan+2])
            t+=6
        return names
    
    width=len(xc)
    mid=spread
    
    g=torch.reshape(g,[inchan,width,width])
    #nchan=uv.shape[0]
    
    
    r=torch.zeros(inchan,mid+1)
    r[:,0]=g[:inchan,mid,mid]**2
    
    for i in range(1,mid+1):
        r[:,i]+=torch.sum(g[:inchan,mid-i:mid+i,mid+i]**2,dim=1)
        r[:,i]+=torch.sum(g[:inchan,mid+i,mid-i:mid+i]**2,dim=1)
        r[:,i]+=torch.sum(g[:inchan,mid-i:mid+i,mid-i]**2,dim=1)
        r[:,i]+=torch.sum(g[:inchan,mid-i,mid-i:mid+i]**2,dim=1)
    for i in range(mid+1):
        r[:,i]=torch.sum(r[:,i:],dim=1)
    if geoflag:   
        ng=(g**2).sum(axis=(1,2),keepdim=True)
        cl2=ng/torch.sqrt(torch.sum(ng**2))
        F=[yc[mid:mid+1],xc[mid:mid+1],\
               r,cl2]
    else:
        F=[yc[mid:mid+1],xc[mid:mid+1],\
               r]
    F=[f.reshape([-1]) for f in F]
    return torch.cat(F,dim=0)
def grad_probe(args):
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    (training_set,training_generator),(val_set,val_generator),(test_set,test_generator),(dataset,glbl_gen)=load_data(data_init,partition,args)
    if isinstance(training_set,climate_data.Dataset2):
        landmasks=training_set.get_masks(glbl=True)
    else:
        landmasks=climate_data.get_land_masks(val_generator)
    device=get_device()
    MASK=landmasks.to(device)
    yc,xc=dataset.coords[0]
    xc=torch.tensor(xc)
    yc=torch.tensor(yc)

    if isinstance(dataset,climate_data.Dataset2):
        MASK=MASK[0,0]
        MASK[MASK==0]=np.nan
        MASK[MASK==MASK]=1
        MASK[MASK!=MASK]=0
    

    device=get_device()
    net.eval()
    for i in range(len(net.nn_layers)-1):
        try:
            net.nn_layers[i].weight.requires_grad=False
        except:
            QQ=1
    geoflag=net.freq_coord
    inchan=net.nn_layers[0].weight.data.shape[1]#net.initwidth
    outchan=net.outwidth
    names=grad_probe_features3([],[],[],inchan,net.spread,listout=True)#,geoflag=geoflag)
    numprobe=names[-1][-1]


    maxsamplecount=1000
    samplecount=np.minimum(len(dataset),maxsamplecount)
    dt=np.maximum(len(dataset)//samplecount,1)
    numbatch=3
    width=net.receptive_field

    chss=np.arange(maxsamplecount)*dt
    np.random.shuffle(chss)
    
    tot=0
    ii=0
    dd=0
    TOTMAX=np.inf
    F=[]
    spread=net.spread
    MASKK=MASK[:-width,:-width]
    stsz=[np.maximum(1,MASKK.shape[i]//70) for i in range(2)]
    MASKK=MASKK[::stsz[0],::stsz[1]]
    KK,LL=np.where(MASKK>0)
    snum=0
    dd=0
    
    samplecount1=20
    GRADS=torch.zeros(samplecount1,len(KK),outchan,inchan,width,width)
    GS=torch.zeros(samplecount,len(KK),numprobe*outchan)

    for i in range(samplecount):
        UV,_,S = dataset[chss[i]]
        for j in range(len(KK)):
            K,L=KK[j]*stsz[0],LL[j]*stsz[1]
            for chan in range(outchan):
                uv=torch.stack([UV[:,K:K+width,L:L+width]],dim=0).to(device)
                uv.requires_grad=True
                output=net.forward(uv)
                x0=output.shape[2]
                x1=output.shape[3]
                m0=(x0-1)//2
                m1=(x1-1)//2
                ou=output[0,chan,m0,m1]
                ou.backward(retain_graph=True)
                g=uv.grad
                uv.grad=None
                uv=uv.to(torch.device("cpu")).detach()
                g=g.to(torch.device("cpu")).detach()
                sample=grad_probe_features3(g,yc[K:K+width],xc[L:L+width],inchan,net.spread)#,geoflag=geoflag)
                GS[i, j,chan*len(sample):(chan+1)*len(sample)]=sample
                if i<samplecount1:
                    GRADS[i,j,chan]=g.reshape([inchan,width,width])
            dd+=1
        if i%args.disp==0:
            print(i,samplecount,flush=True)
            np.save(root+'/grad-probe-data.npy', GS[:i+1])
            if i<samplecount1:
                np.save(root+'/grad-samples.npy', GRADS[:i+1])
        if i==samplecount1:
            np.save(root+'/grad-samples.npy', GRADS)
                


    np.save(root+'/grad-probe-data.npy', GS)


def grad_analysis(args):
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    _,_,_,(dataset,datagen)=load_data(data_init,partition,args)
    
    MASK=climate_data.get_land_masks(datagen)[0,0]
    device=get_device()
    net.eval()
    for i in range(len(net.nn_layers)-1):
        try:
            net.nn_layers[i].weight.requires_grad=False
        except:
            QQ=1
    width=net.receptive_field
    spread=net.spread
    dx=spread
    dy=spread
    
    W=np.reshape(np.arange(spread+1),[-1,1])
    sx=dataset.dimens[1]-width+1
    sy=dataset.dimens[0]-width+1
    
    xx=np.arange(0,sx,dx)
    yy=np.arange(0,sy,dy)
    nx=len(xx)
    ny=len(yy)
    
    
    UV,_,_ = dataset[0]
    nchan=UV.shape[0]
    G=np.zeros((nchan*3,ny*width, nx*width))
    width=net.receptive_field
    maxsamplenum=4*width**2+2
    CN=np.zeros((4*width**2+2,4*width**2+2))
    MS=np.zeros((4*width**2+2,maxsamplenum))
    samplecount=0
    tot=0
    ii=0
    dd=0
    TOTMAX=np.inf
    for i in range(len(dataset)):
        UV,_,_ = dataset[i]
        #for local_batch,nan_mask, _ in datagen:
        #uv, nan_mask = local_batch.to(device),nan_mask.to(device)
        for k in range(ny):
            for l in range(nx):
                dd+=1
                K,L=yy[k],xx[l]
                if MASK[K,L]>0:  
                    uv=torch.stack([UV[:,K:K+width,L:L+width]],dim=0).to(device)
                    uv.requires_grad=True
                    output=net.forward(uv)
                    x0=output.shape[2]
                    x1=output.shape[3]
                    m0=(x0-1)//2
                    m1=(x1-1)//2
                    ou=output[0,0,m0,m1]
                    ou.backward(retain_graph=True)
                    g=uv.grad
                    uv.grad=None
                    uv=uv.to(torch.device("cpu")).detach().numpy()
                    g=g.to(torch.device("cpu")).detach().numpy()
                    g=np.reshape(g,[nchan,width,width])
                    for j in range(nchan):
                        G[j,k*width:(k+1)*width,l*width:(l+1)*width]+=g[j]**2
                        fg=np.abs(np.fft.fftshift(np.fft.fft2(g[j])))
                        G[j+nchan,k*width:(k+1)*width,l*width:(l+1)*width]+=fg**2
                        G[j+2*nchan,k*width:(k+1)*width,l*width:(l+1)*width]+=np.abs(g[j])/np.sum(np.abs(g[j]))
                    
                    ii+=1
                    uv_=np.reshape(uv[0,:2],[-1,1])
                    g_=np.reshape(g[:2],[-1,1])
                    ou=np.reshape(ou.to(torch.device("cpu")).detach().numpy(),[-1,1])
                    vec_=np.concatenate((uv_,ou,g_,np.ones((1,1))),axis=0)
                    CN=CN+vec_@vec_.T
                    tot+=1
                    if samplecount<maxsamplenum:
                        if np.random.randint(2, size=1)[0]==0:
                            MS[:,samplecount:samplecount+1]=vec_
                            samplecount+=1
                    if tot>TOTMAX:
                        break
                else:
                    G[:,k*width:(k+1)*width,l*width:(l+1)*width]=np.nan
                if dd%1000==0:
                    print('\t\t '+str(dd/nx/ny),flush=True)
            if tot>TOTMAX:
                break
                
        print(tot,flush=True)
        with open(root+'/global-grad.npy', 'wb') as f:
            np.save(f, G/tot)
        with open(root+'/global-grad-covariance.npy', 'wb') as f:
            np.save(f, CN/tot)
        with open(root+'/global-grad-samples.npy', 'wb') as f:
            np.save(f, MS)
        if tot>TOTMAX:
            break