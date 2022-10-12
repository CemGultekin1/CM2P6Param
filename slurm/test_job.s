#!/bin/bash
#SBATCH --time=15:00
#SBATCH --array=1-1
#SBATCH --mem=120GB
#SBATCH --job-name=test_job
#SBATCH --output=/scratch/cg3306/climate/climate_research/slurm/test_job_%a.out
#SBATCH --error=/scratch/cg3306/climate/climate_research/slurm/test_job_%a.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/climate_research/slurm/test_job.txt)
module purge
singularity exec --nv --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/train.py $ARGS;\
	"
