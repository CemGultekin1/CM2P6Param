import numpy as np

def data_fourier_analysis(args):
    # WILL NOT WORK BECAUSE OF CHANGES
    net,criterion,logs,PATH0,PATH1,LOG,root=load_from_save(args)

    width=41
    dy=(width-1)//2
    dx=(width-1)//2

    net.spread=dx
    _,_,_,(dataset,datagen)=load_data(net,args)
    MASK=climate_data.get_land_masks(datagen)[0,0]




    sx=dataset.dimens[1]-2*dx-2*dx
    sy=dataset.dimens[0]-2*dx-2*dx

    xx=np.arange(0,sx,dx)
    yy=np.arange(0,sy,dy)
    nx=len(xx)
    ny=len(yy)

    G=np.zeros((8,ny*width, nx*width))


    tot=0

    for i in range(len(dataset)):
        UV,_,SXY = dataset[i]
        #for local_batch,nan_mask, _ in datagen:
        #uv, nan_mask = local_batch.to(device),nan_mask.to(device)
        for k in range(ny):
            for l in range(nx):
                K,L=yy[k],xx[l]
                if MASK[K,L]>0:
                    uv=UV[:,K:K+width,L:L+width].numpy()
                    sxy=SXY[:,K:K+width,L:L+width].numpy()
                    ff=[np.abs(np.fft.fftshift(np.fft.fft2(uv[oo]))) for oo in range(2)]
                    ff=ff+[np.abs(np.fft.fftshift(np.fft.fft2(sxy[oo]))) for oo in range(2)]
                    for j in range(len(ff)):
                        G[j,k*width:(k+1)*width,l*width:(l+1)*width]+=ff[j]**2
                else:
                    G[:,k*width:(k+1)*width,l*width:(l+1)*width]=np.nan
        tot+=1
        print(tot,flush=True)
        NG=torch.tensor(G)
        NG[:4]=NG[:4]/tot
        for k in range(ny):
            for l in range(nx):
                NN=NG[:4,k*width:(k+1)*width,l*width:(l+1)*width]
                MNN=torch.sum(NN,dim=[1,2],keepdim=True)
                NN=NN/MNN
                NG[4:,k*width:(k+1)*width,l*width:(l+1)*width]=NN
        with open('/scratch/cg3306/climate/global-fourier-analysis.npy', 'wb') as f:
            np.save(f, NG.numpy())
