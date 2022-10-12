#!/bin/bash
#SBATCH --time=30:00
#SBATCH --array=1-28
#SBATCH --mem=30GB
#SBATCH --job-name=scalars
#SBATCH --output=/scratch/cg3306/climate/climate_research/slurm/logs/scalars_%a_%A.out
#SBATCH --error=/scratch/cg3306/climate/climate_research/slurm/logs/scalars_%a_%A.err
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/climate_research/slurm/scalars.txt)
module purge
singularity exec --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/scalars.py $ARGS;\
	"