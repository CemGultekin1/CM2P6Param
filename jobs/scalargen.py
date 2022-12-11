import itertools
import os
from jobs.job_body import create_slurm_job
from utils.paths import SLURM_LOGS, SLURM
from data.coords import DEPTHS
JobName = 'scalars'
root = SLURM

NCPU = 8
MEM_PER_CPU = 9
def python_args():
    def givearg(sigma,depth):
        st =  f"--domain global --prefetch_factor 1 --depth {depth} --sigma {sigma} --mode scalars --num_workers {NCPU}"
        return st
    sigmas = [8,12,16]
    depths = [int(d) for d in DEPTHS]
    depths = [depths[0]]
    prods = (sigmas,depths)
    lines = []
    for args in itertools.product(*prods):
        lines.append(givearg(*args))
    njob = len(lines)
    lines = '\n'.join(lines)
    argsfile = JobName + '.txt'
    path = os.path.join(root,argsfile)
    with open(path,'w') as f:
        f.write(lines)
    return njob
def slurm(njob):
    slurmfile =  os.path.join(root,JobName + '.s')
    out = os.path.join(SLURM_LOGS,JobName+ '_%a_%A.out')
    err = os.path.join(SLURM_LOGS,JobName+ '_%a_%A.err')
    create_slurm_job(slurmfile,\
        python_file = 'run/scalars.py',
        time = "3:00:00",array = f"1-{njob}",\
        mem = f"{MEM_PER_CPU*NCPU}GB",job_name = JobName,\
        output = out,error = err,\
        cpus_per_task = str(NCPU),
        nodes = "1",
        ntasks_per_node = "1")

def main():
    njob = python_args()
    slurm(njob)



if __name__=='__main__':
    main()
