def job(argsfile,python_file,**kwargs):
    head = ["#!/bin/bash"]
    intro = [f"#SBATCH --{key.replace('_','-')}={val}" for key,val in kwargs.items()]
    bashline = [
        f"ARGS=$(sed -n \"$SLURM_ARRAY_TASK_ID\"p {argsfile})"
    ]
    bodystart = ["module purge",\
        "singularity exec --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c \"\\"]
    codebody = [
        "source src.sh;",
        f"python3 {python_file} $ARGS;"
    ]
    codebody = ['\t' + cb + '\\' for cb in codebody]
    codebody.append('\t\"')
    return "\n".join(head + intro + bashline + bodystart + codebody)

def create_slurm_job(path,python_file = 'run/train.py',**kwargs):
    argsfile = path.replace('.s','.txt')
    text = job(argsfile,python_file,**kwargs)
    with open(path,'w') as f:
        f.write(text)
