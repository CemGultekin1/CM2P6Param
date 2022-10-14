

import os
from models.search import is_trained, is_viewed
from params import replace_param
from jobs.job_body import create_slurm_job
from utils.arguments import options
from utils.paths import SLURM, SLURM_LOGS

JOBNAME = 'viewjob'
root = SLURM

NCPU = 8


def generate_eval_tasks():
    argsfile = 'trainjob.txt'
    path = os.path.join(root,argsfile)
    file = open(path,'r')
    argslist = file.readlines()
    file.close()
    jobnums = []
    for i,args in enumerate(argslist):
        args = args.split()
        args = replace_param(args,'mode','view')
        argslist[i] = ' '.join(args)

    argsfile = JOBNAME + '.txt'
    path = os.path.join(root,argsfile)
    with open(path,'w') as f:
        f.write('\n'.join(argslist))

    for i,args in enumerate(argslist):
        args = args.split()
        _,modelid = options(args,key = "model")
        if not is_viewed(modelid) and is_trained(modelid):
            jobnums.append(str(i))
    jobarray = ','.join(jobnums)
    slurmfile =  os.path.join(SLURM,JOBNAME + '.s')
    out = os.path.join(SLURM_LOGS,JOBNAME+ '_%a_%A.out')
    err = os.path.join(SLURM_LOGS,JOBNAME+ '_%a_%A.err')
    create_slurm_job(slurmfile,\
        python_file = 'run/view.py',
        time = "1:00:00",array = jobarray,\
        mem = "30GB",job_name = JOBNAME,\
        output = out,error = err,\
        cpus_per_task = str(NCPU),
        nodes = "1",
        gres="gpu:1",
        ntasks_per_node = "1")


def main():
    generate_eval_tasks()

if __name__=='__main__':
    main()
