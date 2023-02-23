import os
from typing import Dict
from models.nets.cnn import adjustcnn
from models.search import is_trained
from params import get_default, replace_param
from jobs.job_body import create_slurm_job
from jobs.taskgen import python_args
from utils.arguments import options
from utils.paths import SLURM, SLURM_LOGS
from data.coords import DEPTHS
TRAINJOB = 'trainjob'
root = SLURM

NCPU = 16
def get_arch_defaults():
    nms = ('widths','kernels','batchnorm','seed','model')
    return {nm : get_default(nm) for nm in nms}
def constant_nparam_model(sigma,kernel_factor = None):
    if kernel_factor is None:
        kernel_factor = 4/sigma
    kwargs = get_arch_defaults()
    widths,kernels = adjustcnn(**kwargs,kernel_factor = kernel_factor,constant_nparam = True)
    return widths,kernels
def getarch(args,**kwargs):
    modelargs,_ = options(args,'model')
    widths,kernels = constant_nparam_model(modelargs.sigma,**kwargs)
    if modelargs.temperature:
        widthin = 3 
    else:
        widthin = 2
    widths[-1] = 2*widthin
    if modelargs.latitude:
        widthin+=2
    widths[0] = widthin
    return tuple(widths),tuple(kernels)
def fix_architecture(args,**kwargs):
    widths,kernels = getarch(args,**kwargs)
    args = replace_param(args,'widths',widths)
    args = replace_param(args,'kernels',kernels)
    return args
def fix_minibatch(args):
    datargs,_ = options(args,key = "data")
    # if datargs.domain == 'global':
    optminibatch = int((datargs.sigma/4)**2*2)
    # else:
    #     optminibatch = int(64*(datargs.sigma/4)**2)
    args = replace_param(args,'minibatch',optminibatch)
    return args
def check_training_task(args):
    runargs,_ = options(args,key = "run")
    if runargs.temperature or runargs.lsrp>0:
        return True
    if runargs.reset_model:
        return False
    _,modelid = options(args,key = "model")
    return is_trained(modelid)

def combine_all(kwargs:Dict[int,dict],base_kwargs):
    argslist = []
    for kwarg in kwargs.values():
        argslist =  argslist + python_args(**kwarg,**base_kwargs)
    return argslist
def generate_training_tasks():
    base_kwargs = dict(
        num_workers = NCPU,
        disp = 50,
        filtering = 'gaussian',
        batchnorm = tuple([1]*7 + [0])
    )
    kwargs = {}
    kwargs[0] = dict(
        lsrp = 0,     
        depth = 0,
        sigma = 4,
        temperature = False,
        lossfun = 'heteroscedastic',
        latitude = False,
        interior = True,
        domain = ['four_regions','global'],
    )
    kwargs[1] = dict(
        lsrp = 0,     
        depth = 0,
        sigma = [4,8,12,16],
        temperature = False,
        lossfun = 'MSE',
        latitude = False,
        domain = ['four_regions','global'],
    )
    kwargs[2] = dict(
        lsrp = [0,1],     
        depth = 0,
        sigma = [4,8,12,16],
        temperature = True,
        lossfun = 'MSE',
        latitude = [False,True],
        domain = ['four_regions','global'],
    )
    kwargs[3] = dict(
        lsrp = [0,1],     
        depth =[int(d) for d in DEPTHS],
        sigma = [8,12,16],
        temperature = True,
        lossfun = 'MSE',
        latitude = [False,True],
        domain = 'global',
    )
    
    argslist = combine_all(kwargs,base_kwargs)
    
    
    for i in range(len(argslist)):
        args = fix_architecture(argslist[i].split())
        args = fix_minibatch(args)
        argslist[i] = ' '.join(args)


    def kernel_size_switched(kernelscale):
        kwargs = dict(
            lsrp = [0,1],
            depth =0,
            sigma = [4,8,12,16],
            temperature = True,
            lossfun = 'MSE',
            latitude = True,
            domain = 'global',
        )
        argslist = python_args(**kwargs,**base_kwargs)
        import numpy as np
        _,idx = np.unique(np.array(argslist),return_index=True)
        argslist = np.array(argslist)
        argslist = argslist[np.sort(idx)].tolist()
        
        for i in range(len(argslist)):
            args = fix_architecture(argslist[i].split(),kernel_factor = kernelscale)
            args = fix_minibatch(args)
            argslist[i] = ' '.join(args)
        return argslist
    kernel_factors = [float(f)/21. for f in [21,15,11,9,7,5,4,3,2,1]]
    argslist_ = []
    for kf in kernel_factors:
        argslist_.extend(kernel_size_switched(kf))
    argslist.extend(argslist_)

    import numpy as np
    _,idx = np.unique(np.array(argslist),return_index=True)
    argslist = np.array(argslist)
    argslist = argslist[np.sort(idx)].tolist()



    
    njobs = len(argslist)
    istrained = []
    for i in range(njobs):
        flag = check_training_task(argslist[i].split())
        print(flag,argslist[i])
        istrained.append(flag)
    jobarray = ','.join([str(i+1) for i in range(njobs) if not istrained[i]])
    njobs = len(argslist)
    lines = '\n'.join(argslist)
    argsfile = TRAINJOB + '.txt'
    path = os.path.join(root,argsfile)
    with open(path,'w') as f:
        f.write(lines)
    slurmfile =  os.path.join(SLURM,TRAINJOB + '.s')
    out = os.path.join(SLURM_LOGS,TRAINJOB+ '_%a_%A.out')
    err = os.path.join(SLURM_LOGS,TRAINJOB+ '_%a_%A.err')
    create_slurm_job(slurmfile,\
        time = "12:00:00",array = jobarray,\
        mem = "150GB",job_name = TRAINJOB,\
        output = out,error = err,\
        cpus_per_task = str(NCPU),
        nodes = "1",
        gres="gpu:1",
        ntasks_per_node = "1")


def main():
    generate_training_tasks()

if __name__=='__main__':
    main()
