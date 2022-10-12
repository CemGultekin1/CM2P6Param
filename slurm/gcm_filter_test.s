#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00
#SBATCH --mem=24GB
#SBATCH --job-name=savelsrp
#SBATCH --output=gcm.out
#SBATCH --error=gcm.err

module purge
singularity exec --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
        /bin/bash -c "\
        source src.sh;\
        python3 dataloader_test.py;\
        "
