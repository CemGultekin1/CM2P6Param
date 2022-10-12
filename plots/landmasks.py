import sys
from typing import List
from data.datasets import Dataset
from data.load import get_loaders, load_normalization_scalars,depthvals
from plots.projections import cartopy_plot
import numpy as np

def pre_processing(datargs,recfield,*datasets:List[Dataset]):
    load_normalization_scalars(datargs,datasets[0])
    assert datasets[0].inscalars is not None
    datasets[0].set_receptive_field(recfield)
    for i in range(1,len(datasets)):
        datasets[i].receive_scalars(datasets[0])
    return datasets

def landmasks():
    def get_mask(depthval,org_field_view=21,sigma=8):
        field_view = int(org_field_view*4/sigma/2)*2+1
        spread = field_view//2
        assert spread>0
        datargs = f'--domain global --depth {depthval} --sigma {sigma}'.split(' ')
        (training_set,_),_,_=get_loaders(datargs)
        tid = 0
        domid = 0
        training_set.set_receptive_field(field_view)
        _,_,mask,lat,lon = training_set.get_pure_data(domid,tid)
        mask[mask==0] = np.nan
        title = f'depth={depthval} meters'
        return mask[0,::-1],lon,lat[spread:-spread],title
    figargs = {i:[] for i in range(4)}
    for depthval in depthvals:
        returns = get_mask(depthval)
        for j in range(len(returns)):
            figargs[j].append(returns[j])
    figargs = list(figargs.values())
    n = len(figargs[0])
    kwargs = [{"colorbar":False} for _ in range(n)]
    figsize = (25,15)
    suptitle = f'land mask for coarse-graining={8}, kernel={11}x{11}'
    address = "/scratch/cg3306/climate/plots/datapeek/"
    import os
    os.makedirs(address)
    filename = os.path.join(address,'landmasks.png')
    cartopy_plot(*figargs,3,3,figsize,suptitle,filename,kwargs)

def main():
    landmasks()


if __name__=='__main__':
    main()
