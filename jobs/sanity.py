

import itertools
import os
from data.scalars import scalars_exist
from models.nets.cnn import adjustcnn
from jobs.job_body import create_slurm_job
from utils.arguments import options
from utils.paths import  SLURM, SLURM_LOGS

DEPTHJOB = 'sanity'
root = SLURM

NCPU = 12
def get_arch_defaults():
    args = "--sigma 4".split()
    archargs,_ = options(args,key = "arch")
    return archargs.widths,archargs.kernels,archargs.batchnorm,archargs.skipconn
def constant_nparam_model(sigma):
    widths,kernels,batchnorm,skipconn = get_arch_defaults()
    widths,kernels = adjustcnn(widths,kernels,batchnorm,skipconn,0,kernel_factor = 4/sigma,constant_nparam = True)
    return widths,kernels

def python_args():
    parts = [1,1]
    def givearg(sigma,latitude,linsupres,depth):
        st =  f"--sigma {sigma} --depth {depth} --disp 1 --parts {' '.join([str(p) for p in parts])} --domain global --normalization standard --lossfun MSE --latitude {latitude} --linsupres {linsupres} --temperature True --num_workers {NCPU*2}"
        widths,kernels = constant_nparam_model(sigma)
        widthin = 3 
        if latitude:
            widthin+=2
        widths[0] = widthin
        widths[-1] = 6

        st = f"{st}  --widths  {' '.join([str(k) for k in widths])}"
        st = f"{st} --kernels  {' '.join([str(k) for k in kernels])}"
        minibatch = 1
        st = f"{st} --rerun True --relog True --sanity True --minibatch {minibatch}"
        return st

    sigma = [8]
    linsupres = [False,]
    latitude = [False,]
    depth = [5,]
    prods = (sigma,linsupres,latitude,depth)
    lines = []
    for args in itertools.product(*prods):
        lines.append(givearg(*args))

    for i,line in enumerate(lines):
        flag,i = scalars_exist(line.split())
        if not flag:
            print(line)
    njobs = len(lines)
    lines = '\n'.join(lines)

    argsfile = DEPTHJOB + '.txt'
    path = os.path.join(root,argsfile)
    with open(path,'w') as f:
        f.write(lines)
    return njobs
def slurm(njobs):
    slurmfile =  os.path.join(SLURM,DEPTHJOB + '.s')
    out = os.path.join(SLURM_LOGS,DEPTHJOB+ '_%a_%A.out')
    err = os.path.join(SLURM_LOGS,DEPTHJOB+ '_%a_%A.err')
    create_slurm_job(slurmfile,\
        time = "1:00:00",array = f"1-{njobs}",\
        mem = "30GB",job_name = DEPTHJOB,\
        output = out,error = err,\
        cpus_per_task = str(NCPU),
        nodes = "1",
        ntasks_per_node = "1")


def main():
    njobs = python_args()
    slurm(njobs)

if __name__=='__main__':
    main()
