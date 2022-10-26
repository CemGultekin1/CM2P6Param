
import itertools
import os
from jobs.job_body import create_slurm_job
from utils.paths import SLURM, SLURM_LOGS

JOBNAME = 'datagen'
root = SLURM

NCPU = 20

def python_args():
    def givearg(sigma,depth):
        st =  f"--minibatch 1 --prefetch_factor 1 --depth {depth} --sigma {sigma} --mode data --num_workers {NCPU}"
        return st
    sigmas = [4,8,12,16]
    depths = [0,5]
    prods = (sigmas,depths)
    lines = []
    for args in itertools.product(*prods):
        lines.append(givearg(*args))
    njob = len(lines)
    lines = '\n'.join(lines)

    argsfile = JOBNAME + '.txt'
    path = os.path.join(root,argsfile)
    with open(path,'w') as f:
        f.write(lines)
    return njob
def slurm(njob):
    slurmfile =  os.path.join(SLURM,JOBNAME + '.s')
    out = os.path.join(SLURM_LOGS,JOBNAME+ '_%a_%A.out')
    err = os.path.join(SLURM_LOGS,JOBNAME+ '_%a_%A.err')
    create_slurm_job(slurmfile,\
        python_file = 'run/datagen.py',
        time = "48:00:00",array = f"1-{njob}",\
        mem = f"{int(NCPU*5)}GB",job_name = JOBNAME,\
        output = out,error = err,\
        cpus_per_task = str(NCPU),
        nodes = "1",
        ntasks_per_node = "1")

def main():
    njob = python_args()
    slurm(njob)



if __name__=='__main__':
    main()
