

import itertools
import os
from data.save import scalars_exist
from models.nets.cnn import adjustcnn
from slurm.job_body import create_slurm_job
from utils.arguments import options
from utils.paths import SLURM

DEPTHJOB = 'deep_train'
root = SLURM

NCPU = 8
def get_arch_defaults():
    args = "--sigma 4".split()
    archargs,_ = options(args,key = "arch")
    return archargs.widths,archargs.kernels,archargs.batchnorm,archargs.skipconn
def constant_nparam_model(sigma):
    widths,kernels,batchnorm,skipconn = get_arch_defaults()
    widths,kernels = adjustcnn(widths,kernels,batchnorm,skipconn,0,kernel_factor = 4/sigma,constant_nparam = True)
    return widths,kernels

def python_args():
    parts = [2,2]
    def givearg(sigma,latitude,linsupres,depth):
        st =  f"--sigma {sigma} --depth {depth} --parts {' '.join([str(p) for p in parts])} --domain global --normalization standard --lossfun MSE --latitude {latitude} --linsupres {linsupres} --temperature True --num_workers {NCPU*2}"
        widths = [3,128,64,32,32,32,32,32,6]
        widths,kernels = constant_nparam_model(sigma)
        widthin = 3 # temperature by default
        if latitude:
            widthin+=2
        widths[0] = widthin
        widths[-1] = 6

        st = f"{st}  --widths  {' '.join([str(k) for k in widths])}"
        st = f"{st} --kernels  {' '.join([str(k) for k in kernels])}"
        minibatch = int(parts[0]*parts[1]*2*sigma/4)
        st = f"{st} --rerun False --relog False --minibatch {minibatch}"
        return st

    sigma = [8]
    linsupres = [False,True]
    latitude = [False,True]
    depth = [5,55,110,181,330,1497]
    prods = (sigma,linsupres,latitude,depth)
    lines = []
    for args in itertools.product(*prods):
        lines.append(givearg(*args))

    for i,line in enumerate(lines):
        flag,i = scalars_exist(line.split())
        if not flag:
            print(line)
            
    lines = '\n'.join(lines)
    
    argsfile = DEPTHJOB + '.txt'
    path = os.path.join(root,argsfile)
    with open(path,'w') as f:
        f.write(lines)
def slurm():
    slurmfile =  os.path.join(root,DEPTHJOB + '.s')
    out = os.path.join(root,'logs',DEPTHJOB+ '_%a.out')
    err = os.path.join(root,'logs',DEPTHJOB+ '_%a.err')
    create_slurm_job(slurmfile,\
        time = "36:00:00",array = "1-12",\
        mem = "120GB",job_name = DEPTHJOB,\
        output = out,error = err,\
        cpus_per_task = str(NCPU),
        nodes = "1",
        gres="gpu:1",
        ntasks_per_node = "1")


def main():
    python_args()
    slurm()

if __name__=='__main__':
    main()