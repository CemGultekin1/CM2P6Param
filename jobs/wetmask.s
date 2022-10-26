#!/bin/bash
#SBATCH --time=15:00
#SBATCH --array=1
#SBATCH --mem=15GB
#SBATCH --job-name=wetmask
#SBATCH --output=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/wetmask_%a_%A.out
#SBATCH --error=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/wetmask_%a_%A.err
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
module purge
singularity exec --nv --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 test.py;\
	"