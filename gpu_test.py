import sys
from data.high_res import HighResCm2p6
from data.load import dataset_arguments
from data.paths import get_preliminary_low_res_data_location
from data.no_torch_load import get_data
from utils.arguments import options
from utils.slurm import flushed_print
import xarray as xr
import time

def main():
    # print('hi')
    datargs = sys.argv[1:]
    _args,_kwargs = dataset_arguments(datargs)
    ds = HighResCm2p6(*_args, **_kwargs)
    sc1 = time.time()
    x = ds[0]
    sc2 = time.time()
    dt = [sc2 - sc1]
    print(dt)
    for i in range(1,11):
        sc1 = time.time()
        x = ds[i]
        sc2 = time.time()
        dt.append(sc2 - sc1)
        print(dt)
    return


if __name__=='__main__':
    main()