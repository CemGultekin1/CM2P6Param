import sys
from data.paths import get_filename
from data.load import get_data
from utils.arguments import options
from utils.slurm import flushed_print
from utils.xarray import fromnumpydict
import xarray as xr



def run():
    datargs = sys.argv[1:]
    generator,= get_data(datargs,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('all',))
    datargs,_ = options(datargs,key = "data")
    filename = get_filename(datargs.sigma,datargs.depth >1e-3,False)
    dst = None
    initflag = False
    for data_vars,coords in generator:
        ds = fromnumpydict(data_vars,coords)
        flushed_print(ds.time.values[0],ds.depth.values)
        if dst is not None:
            if ds.time.values[0] != dst.time.values[0]:
                chk = {k:len(dst[k]) for k in list(dst.coords)}
                if not initflag:
                    dst = dst.chunk(chunks=chk)
                    dst.to_zarr(filename,mode='w')
                    initflag = True
                else:
                    _coords = dst.coords
                    timely = {key:val for key,val in dst.data_vars.items() if 'time' in val.dims}
                    dst = xr.Dataset(data_vars = timely,coords = _coords)
                    dst = dst.chunk(chunks=chk)
                    dst.to_zarr(filename,mode='a',append_dim = 'time')
                dst = None
        if dst is None:
            dst = ds
        else:
            dst = xr.merge([dst,ds])

if __name__=='__main__':
    run()
