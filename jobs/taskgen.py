import itertools
from models.nets.cnn import adjustcnn
from utils.arguments import options
from utils.paths import SLURM

def get_arch_defaults():
    args = "--sigma 4".split()
    archargs,_ = options(args,key = "arch")
    return archargs.widths,archargs.kernels,archargs.batchnorm,archargs.skipconn
def constant_nparam_model(sigma):
    widths,kernels,batchnorm,skipconn = get_arch_defaults()
    widths,kernels = adjustcnn(widths,kernels,batchnorm,skipconn,0,kernel_factor = 4/sigma,constant_nparam = True)
    return widths,kernels

def python_args(**kwargs):
    def givearg(inds):
        args = []
        for i,(key,vals) in zip(inds,kwargs.items()):
            if isinstance(vals,list):
                val = vals[i]
            else:
                val = vals
            args.append(f'--{key}')
            args.append(str(val))
        return ' '.join(args)
    indices = []
    for val in kwargs.values():
        if isinstance(val,list):
            indices.append(len(val))
        else:
            indices.append(1)
    indices = [range(k) for k in indices] 
    args = []
    for prodind in itertools.product(*indices):
        args.append(givearg(prodind))
    return args
        