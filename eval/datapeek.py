from typing import List
from data.datasets import Dataset
from data.load import get_loaders, load_normalization_scalars,depthvals
from plots.projections import save_fig
import numpy as np

def pre_processing(datargs,recfield,*datasets:List[Dataset]):
    load_normalization_scalars(datargs,datasets[0])
    assert datasets[0].inscalars is not None
    datasets[0].set_receptive_field(recfield)
    for i in range(1,len(datasets)):
        datasets[i].receive_scalars(datasets[0])
    return datasets



def peek(datargs):
    (training_set,_),_,_=get_loaders(datargs)
    tid = 0
    domid = 0
    training_set.set_receptive_field(1)
    infield,outfield,mask,lat,lon = training_set.get_pure_data(domid,tid)
    # infield = infield[:,:,10:-10]
    infield[infield==0] = np.nan
    mask[mask==0] = np.nan
    outfield = outfield*mask

    lat = lat[::-1]



    n = infield.shape[0]
    lons = [lon]*n*2
    lats = [lat]*n*2
    infields = [f for f in infield]

    # lons = lons + [lon[10:-10]]*n
    # lats = lats + [lat[10:-10]]*n

    outfields = [f for f in outfield]
    fields = infields + outfields
    for f,lon,lat in zip(fields,lons,lats):
        print(f.shape,lon.shape,lat.shape)

    kwargs = [{} for _ in range(len(fields))]
    figsize = (40,30)
    save_fig(fields,lons,lats,2,3,figsize,'test.png',kwargs)


def main():
    # datargs = '--domain custom --temperature True --depth 0 --sigma 8 --linsupres True'.split(' ')
    # peek(datargs)
    data_masks()


if __name__=='__main__':
    main()
