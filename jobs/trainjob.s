#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --array=0,1,2,3
#SBATCH --mem=120GB
#SBATCH --job-name=trainjob
#SBATCH --output=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/trainjob_%a_%A.out
#SBATCH --error=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/trainjob_%a_%A.err
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/CM2P6Param/jobs/trainjob.txt)
module purge
singularity exec --nv --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/train.py $ARGS;\
	"