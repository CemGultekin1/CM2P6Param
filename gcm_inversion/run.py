
from gcm_inversion.collect_weights import GCM_weights
from gcm_inversion.compressing import Compressor
import numpy as np
from utils.xarray import plot_ds
import xarray as xr
def to_data_array(f,sigma):
    return xr.DataArray(
            data = f,
            dims = ['lat','lon'],
            coords = dict(
                lat = np.arange(-2*sigma,3*sigma + 1),lon = np.arange(-2*sigma,3*sigma + 1)
            )
        )
def main():
    sigma= 4
    gcmw = GCM_weights(sigma)
    buff_size = 2
    totalsize = 4
    cmp = Compressor(buff_size)
    np.random.seed(0)
    wetinds = np.random.choice(np.arange(gcmw.nwet),size = totalsize//2,replace = False)
    mixedinds = np.random.choice(np.arange(gcmw.nmixed),size = totalsize//2,replace = False)
    wetlocs = gcmw.wet_indices[wetinds]
    mixlocs = gcmw.mixed_indices[mixedinds]
    def feed_filters(wetinds,navalued):
        lfz = dict()
        for z,wi in enumerate(wetinds):
            lf = gcmw.loc_filters(wi,navalued=navalued)
            lfz[str(tuple(wi))] = to_data_array(lf,sigma)
            print(z,wi,lf.shape)
            cmp.process(lf,wi)
        print()
        return lfz
    lfz = feed_filters(wetlocs[:buff_size//2],False)
    plot_ds(lfz,'loc_filters_0')
    lfz = feed_filters(mixlocs[:buff_size//2],True)
    plot_ds(lfz,'loc_filters_1')
    lfz = feed_filters(wetlocs[buff_size//2:totalsize//2],False)
    plot_ds(lfz,'loc_filters_2')
    lfz = feed_filters(mixlocs[buff_size//2:totalsize//2],True)
    plot_ds(lfz,'loc_filters_3')


    lfs = dict()
    
    for i in range(len(cmp.address)):
        ad = tuple(cmp.address[i])
        f = cmp.rebuild_filter(i)
        ff = gcmw.apply_mask(f,ad)
        tag = 'mix' if np.any(np.isnan(ff)) else 'wet'
        lfs[f"{str(ad)}_{tag}"] = to_data_array(f,sigma)
    plot_ds(lfs,'loc_filters')
    return

if __name__=='__main__':
    main()
