#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=fman
#SBATCH --output=folder_change.out
#SBATCH --error=folder_change.err

module purge
singularity exec --overlay .ext3_:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
        /bin/bash -c "\
        source src.sh;\
        python3 dataloader_test.py;\
        "
