from data.load import get_data, load_xr_dataset
from transforms.coarse_grain import coarse_graining_2d_generator
import  matplotlib.pyplot as plt
from transforms.grids import trim_expanded_longitude
from transforms.subgrid_forcing import subgrid_forcing

def main():
    
    args = f'--mode data --sigma {sigma}'.split()
    ds = get_data(args,torch_flag = False,data_loaders = False)
    ds.save_filters()
    # import xarray as xr
    # for sigma in [4,8,12,16]:
    #     sfd = xr.open_dataset(f"subgrid_forcings_{sigma}.nc")
    #     print(sfd,'\n\n\n\n')
    # return

    # args = '--mode data --depth 55'.split()
    # ds = load_xr_dataset(args)

    # ilon = slice(1000,1600)
    # ilat = slice(1000,1600)
    # ds = ds.isel(ulat = ilat,ulon = ilon).isel(tlat = ilat,tlon =ilon )
    # sigma = 4
    # lrd = LowResDataset(ds,sigma)
    # sfd = lrd[0]
    # print(sfd)
    # return

    # for sigma in [4,8,12,16]:
    #     lrd = LowResDataset(ds,sigma)
    #     sfd = lrd[0]
    #     for key in sfd.data_vars:
    #         sfd[key].plot()
    #         plt.savefig(f"{key}_{sigma}.png")
    #         plt.close()
    #     sfd.to_netcdf(path = f"subgrid_forcings_{sigma}.nc")


    return 
    ds = ds.isel(time = 0).isel(ulat = slice(1000,1600),ulon = slice(1000,1600)).isel(tlat = slice(1000,1600),tlon = slice(1000,1600))
    u = ds.u
    u = u.rename({'ulat':'lat','ulon':'lon'})
    u.name = 'u'

    v = ds.v
    v = v.rename({'ulat':'lat','ulon':'lon'})
    v.name = 'v'

    T = ds.T
    T = T.rename({'tlat':'lat','tlon':'lon'})
    T.name = 'T'
    
    cgu = coarse_graining_2d_generator(u,sigma=4,wetmask = True)
    cgt = coarse_graining_2d_generator(T,sigma=4,wetmask = True)

    sgs = subgrid_forcing(u,v,T,cgu,cgt)

    
    

if __name__=='__main__':
    main()
