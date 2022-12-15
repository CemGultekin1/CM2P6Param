import xarray as xr
import os

def main():
    root = '/scratch/zanna/data/cm2.6/'
    filename = 'coarse_4_surface_.zarr'
    path = os.path.join(root,filename)
    ds = xr.open_zarr(path)
    import matplotlib.pyplot as plt
    # u = ds.u.isel(time = 0)
    # u.plot()
    # plt.savefig('time_seq_N30_W60.png')
    # plt.close()
    # return
    t0 = 4
    u = ds.Su.sel(lat = 30,lon = -60,method='nearest').isel(time = range(t0,t0+300))*1e7
    print(u)
    
    uval = u.values

    fig,axs = plt.subplots(2,1,figsize = (15,15))
    axs[0].plot(uval)
    axs[0].set_ylabel('$1e^{-7}m/s^2$')
    axs[0].set_ylim([-5.1,3.1])
    u = ds.Su.sel(lat = -20,lon = -104,method='nearest').isel(time = range(t0,t0+300))*1e7
    uval = u.values
    axs[1].plot(uval)
    axs[1].set_ylabel('$1e^{-7}m/s^2$')
    axs[1].set_ylim([-0.31,0.21])
    plt.savefig('time_seq.png')
    plt.close()


if __name__ =='__main__':
    main()