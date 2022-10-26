




from data.gcm_forcing import SingleDomain


def main(): 
    datargs = '--sigma 4'.split()
    from data.load import load_xr_dataset
    ds = load_xr_dataset(datargs)
    sdm = SingleDomain(ds,4,half_spread = 5,var_grouping = [('u v T'.split()),('Su Sv ST lsrp_res_Su lsrp_res_Sv lsrp_res_ST'.split())])
    UF = sdm[0]
    print(UF)
    return
    datargs = '--sigma 4'.split()
    from data.paths import get_filename
    from utils.arguments import options
    datargs,_ = options(datargs,key = "data")
    filename = get_filename(datargs.sigma,datargs.depth >1e-3,False)
    import xarray as xr
    ds = xr.open_zarr(filename)
    ds = ds.isel(time = 0,depth = 0)
    print(ds)
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(3,4,figsize= (30,30))
    for i,key in enumerate('u v T'.split()):
        ds[key].plot(ax = axs[i,0])
        ds[f'S{key}'].plot(ax = axs[i,1])
        ds[f'lsrp_S{key}'].plot(ax = axs[i,2])
        err = ds[f'S{key}'] - ds[f'lsrp_S{key}']
        err.plot(ax = axs[i,3])
    fig.savefig('coarse_fig.png')

    return
    

if __name__=='__main__':
    main()
