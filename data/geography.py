import numpy as np
import torch




def get_land_masks(val_gen):
    val_gen.dataset.no_more_mask_flag=False
    ii=0
    for local_batch,local_masks,_ in val_gen:
        if ii==0:
            bsize=local_batch.shape[0]
            masks=torch.zeros((val_gen.dataset.num_domains//bsize + 1)*bsize,local_masks.shape[1],local_masks.shape[2],local_masks.shape[3])
        masks[ii:ii+local_masks.shape[0]]=local_masks
        ii+=local_masks.shape[0]
        if ii>=val_gen.dataset.num_domains:
            break
    masks=masks[0:val_gen.dataset.num_domains]
    val_gen.dataset.no_more_mask_flag=True
    return masks


def physical_forces(x):
    dudy=x[:,:2,2:,1:-1]-x[:,:2,:-2,1:-1]
    dudx=x[:,:2,1:-1,2:]-x[:,:2,1:-1,:-2]
    x=x[:,:,1:-1,1:-1]
    u_=x[:,0:1]
    v_=x[:,1:2]
    x=torch.cat([x,dudy,dudx,u_*dudy,v_*dudy,u_*dudx,v_*dudx],dim=1)
    return x


# def zigzag_freq(n,m,f0,df,d=1,reps=100):
#     x=np.linspace(0,m,n)
#     for _ in range(m):
#         x=np.abs(1-np.abs(1-x))
#     x=x**d*df+f0
#     x=np.cumsum(x)
#     x=x/x[-1]*2*np.pi*reps
#     return x

# def sigmoid_freq(n,f0,df,d=20,reps=100):
#     x=np.linspace(-1,1,n)
#     x=1/(1+np.exp(-x*d))
#     x=x*df+f0
#     x=np.cumsum(x)
#     x=x/x[-1]*2*np.pi*reps
#     return x
# def geographic_features(y,x):
#     lat1=zigzag_freq(len(y),2,(30*645)//len(y),40,d=1,reps=55)
#     lat2=sigmoid_freq(len(y), (30*645)//len(y),30,d=15,reps=55)
#     lng1=zigzag_freq(len(x),2,(30*645)//len(y),50,d=1,reps=70)
#     lng2=zigzag_freq(len(x),4,(30*645)//len(y),50,d=1,reps=70)
#     return lat1, lat2, lng1, lng2



def hat_freq(n,span):
    p0=1/2
    p1=4
    m=2
    x=np.linspace(0,m,n)
    for _ in range(m):
        x=np.abs(1-np.abs(1-x))
    Pmin=span*p0
    Pmax=span*p1

    Fmin=1/Pmax
    Fmax=1/Pmin
    dF=(Fmax-Fmin)
    x=x*dF+Fmin
    x=np.cumsum(x)
    return x*2*np.pi

def sigmoid_freq(n,span):
    p0=1/2
    p1=4
    d=20
    
    x=np.linspace(-1,1,n)
    x=1/(1+np.exp(-x*d))
    
    Pmin=span*p0
    Pmax=span*p1

    Fmin=1/Pmax
    Fmax=1/Pmin
    dF=(Fmax-Fmin)
    
    x=x*dF+Fmin
    x=np.cumsum(x)
    return x*2*np.pi
def frequency_encoded_latitude(n,span):
    lat1=hat_freq(n,span)
    lat2=sigmoid_freq(n,span)
    return lat1, lat2





def main():
    n = 645
    span = 21
    lat1,lat2 = frequency_encoded_latitude(n,span)
    import matplotlib.pyplot as plt
    plt.plot(np.cos(lat1))
    plt.savefig('lat1.png')
    plt.close()

    plt.plot(np.cos(lat2))
    plt.savefig('lat2.png')
    plt.close()

if __name__=='__main__':
    main()