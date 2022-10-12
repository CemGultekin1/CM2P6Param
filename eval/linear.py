def data_covariance(args):
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    _,_,_,(dataset,datagen)=load_data(data_init,partition,args)

    MASK=climate_data.get_land_masks(datagen)[0,0]
    device=get_device()
    net.eval()
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
    G=np.zeros((nchan*2,ny*width, nx*width))
    width=net.receptive_field
    COV=np.zeros((2*width**2+1,2*width**2+1))
    samplecount=0
    tot=0
    TOTMAX=np.inf
    for i in range(len(dataset)):
        UV,_,_ = dataset[i]
        #for local_batch,nan_mask, _ in datagen:
        #uv, nan_mask = local_batch.to(device),nan_mask.to(device)
        for k in range(ny):
            for l in range(nx):
                K,L=yy[k],xx[l]
                if MASK[K,L]>0:
                    uv=torch.reshape(UV[:2,K:K+width,L:L+width],(1,-1))
                    uv=torch.cat([uv,torch.ones(1,1)],dim=1)
                    COV=COV+(uv.T@uv).numpy()
                    tot+=1
                    if tot>TOTMAX:
                        break
            if tot>TOTMAX:
                break
        if tot>TOTMAX:
            break
        if i%10==0:
            print('\t\t '+str(tot/nx/ny/len(dataset)),flush=True)
        with open('/scratch/cg3306/climate/data-covariance.npy', 'wb') as f:
            np.save(f, COV/tot)

def projection_analysis(args):
    model_id=int(args.model_id)%4
    sigma_id=int(args.model_id)//4
    sigma_vals=[4,8,12,16]
    sigma=sigma_vals[sigma_id]
    data_root='/scratch/zanna/data/cm2.6/coarse-'
    raw_add=['3D-data-sigma-'+str(sigma)+'.zarr','surf-data-sigma-'+str(sigma)+'.zarr']
    raw_co2=['1pct-CO2-'+ss for ss in raw_add]
    raw_add=raw_add+raw_co2
    raw_data_address=raw_add[model_id]
    ds_data=xr.open_zarr(data_root+raw_data_address)
    MSELOC='/scratch/cg3306/climate/projection_analysis/'+raw_data_address.replace('.zarr','')+'MSE.npy'
    SC2LOC='/scratch/cg3306/climate/projection_analysis/'+raw_data_address.replace('.zarr','')+'SC2.npy'
    noutsig=3
    names='Su Sv ST'.split()
    MSE=torch.zeros(noutsig,ds_data.Su.shape[-2], ds_data.Su.shape[-1])
    SC2=torch.zeros(noutsig,ds_data.Su.shape[-2], ds_data.Su.shape[-1])
    print(MSELOC)
    T=len(ds_data.time.values)
    for i in range(T):
        uv=ds_data.isel(time=i)
        for j in range(len(names)):
            Sxy=uv[names[j]].values
            output=Sxy-uv[names[j]+'_r'].values
            SC2[j]+=Sxy**2
            MSE[j]+=(Sxy-output)**2
        if i%50==0:
            with open(MSELOC, 'wb') as f:
                np.save(f, MSE/(i+1))
            with open(SC2LOC, 'wb') as f:
                np.save(f, SC2/(i+1))
            print('\t #'+str(i),flush=True)
    with open(MSELOC, 'wb') as f:
        np.save(f, MSE/(i+1))
    with open(SC2LOC, 'wb') as f:
        np.save(f, SC2/(i+1))
