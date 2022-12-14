#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --array=1-1
#SBATCH --mem=30GB
#SBATCH --job-name=sanity
#SBATCH --output=/scratch/cg3306/climate/CM2P6Param/slurm_logs/sanity_%a_%A.out
#SBATCH --error=/scratch/cg3306/climate/CM2P6Param/slurm_logs/sanity_%a_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/CM2P6Param/slurm_jobs/sanity.txt)
module purge
singularity exec --nv --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/train.py $ARGS;\
	"