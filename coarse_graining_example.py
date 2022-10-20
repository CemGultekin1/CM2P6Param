from data.gcm_forcing import SingleDomain
from data.load import get_high_res_data_location, preprocess_dataset
from utils.arguments import options    
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
def main():
    args = '--sigma 8 --depth 330'.split()
    datargs, _ = options(args,key = "data")
    hres_path = get_high_res_data_location(args)
    print(hres_path)
    hresdata = xr.open_zarr(hres_path,consolidated=False).isel(st_edges_ocean = datargs.sigma)
    hresdata = preprocess_dataset(args,hresdata)
    boundaries = (-40,40,-40,40)
    sd = SingleDomain(hresdata,datargs.sigma,boundaries = boundaries,coarse_grain_needed = True,wetmask = False)
    sd1 = SingleDomain(hresdata,datargs.sigma,boundaries = boundaries,coarse_grain_needed = True,wetmask = True)
    hresu,_,_ = sd.get_grid_fixed_hres(0)
    hresmask = xr.where(np.isnan(hresu),1,0)
    UF = sd[0]
    UF1 = sd1[0]
    u = UF['fields'].u
    Su = UF['forcings'].u
    fim = UF['field_wetmask']>0
    fom = UF['forcing_wetmask']

    u1 = UF1['fields'].u
    Su1 = UF1['forcings'].u
    fim1 = UF1['field_wetmask']>0
    fom1 = UF1['forcing_wetmask']

    fig,axs = plt.subplots(2,5,figsize = (60,30))
    hresu.plot(ax = axs[0,0])
    u.plot(ax = axs[0,1])
    fim.plot(ax = axs[0,2])
    Su.plot(ax = axs[0,3])
    fom.plot(ax = axs[0,4])


    hresmask.plot(ax = axs[1,0])
    u1.plot(ax = axs[1,1])
    fim1.plot(ax = axs[1,2])
    Su1.plot(ax = axs[1,3])
    fom1.plot(ax = axs[1,4])
    fig.savefig('dummy.png')
    plt.close()
    
if __name__ == '__main__':
    main()