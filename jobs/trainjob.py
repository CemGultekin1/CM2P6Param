import os
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

NCPU = 15
def get_arch_defaults():
    nms = ('widths','kernels','batchnorm','skipconn')
    return (get_default(nm) for nm in nms)
def constant_nparam_model(sigma):
    widths,kernels,batchnorm,skipconn = get_arch_defaults()
    widths,kernels = adjustcnn(widths,kernels,batchnorm,skipconn,0,kernel_factor = 4/sigma,constant_nparam = True)
    return widths,kernels
def getarch(args):
    modelargs,_ = options(args,'model')
    widths,kernels = constant_nparam_model(modelargs.sigma)
    if modelargs.temperature:
        widthin = 3 
    else:
        widthin = 2
    widths[-1] = 2*widthin
    if modelargs.latitude:
        widthin+=2
    widths[0] = widthin
    return tuple(widths),tuple(kernels)
def fix_architecture(args):
    widths,kernels = getarch(args)
    args = replace_param(args,'widths',widths)
    args = replace_param(args,'kernels',kernels)
    return args
def fix_minibatch(args):
    datargs,_ = options(args,key = "data")
    if datargs.domain == 'global':
        optminibatch = int(datargs.parts[0]*datargs.parts[1]*datargs.sigma/4*2)
    else:
        optminibatch = 64
    args = replace_param(args,'minibatch',optminibatch)
    return args
def check_training_task(args):
    runargs,_ = options(args,key = "run")
    if runargs.rerun:
        return False
    _,modelid = options(args,key = "model")
    return is_trained(modelid)

def generate_training_tasks():
    kwargs = dict(
        lsrp = [0],     
        depth = [0],
        sigma = [4,8,12,16],
        temperature = False,
        lossfun = 'MSE',
        latitude = [False],
        domain = ['four_regions','global'],
        num_workers = NCPU,
        disp = 100,
        rerun = True,
        relog = True
    )
    argslist = python_args(**kwargs)


    kwargs = dict(
        lsrp = [0,1,2],     
        depth = [0],
        sigma = [4,8,12,16],
        temperature = True,
        lossfun = 'MSE',
        latitude = [False,True],
        domain = ['four_regions','global'],
        num_workers = NCPU*2,
        disp = 100,
        rerun = True,
        relog = True
    )
    argslist = argslist + python_args(**kwargs)

    kwargs = dict(
        lsrp = [0,1,2],     
        depth =[int(d) for d in DEPTHS],
        sigma = [8],
        temperature = True,
        lossfun = 'MSE',
        latitude = [False,True],
        domain = 'global',
        num_workers = NCPU*2,
        disp = 100,
        rerun = True,
        relog = True
    )
    argslist = argslist + python_args(**kwargs)
    import numpy as np
    _,idx = np.unique(np.array(argslist),return_index=True)
    argslist = np.array(argslist)
    argslist = argslist[np.sort(idx)].tolist()
    
    for i in range(len(argslist)):
        args = fix_architecture(argslist[i].split())
        args = fix_minibatch(args)
        argslist[i] = ' '.join(args)
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
        time = "24:00:00",array = jobarray,\
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
