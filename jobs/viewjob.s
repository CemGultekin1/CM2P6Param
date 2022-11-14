#!/bin/bash
#SBATCH --time=45:00
#SBATCH --array=2,3,4,5,6,7,8,13,14,15,16,17,18,19,20,21,22,23,24,29,30,31,32,33,34,35,36,37,38,39,40,45,46,47,48,49,50,51,52,53,54
#SBATCH --mem=12GB
#SBATCH --job-name=viewjob
#SBATCH --output=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/viewjob_%a_%A.out
#SBATCH --error=/scratch/cg3306/climate/CM2P6Param/saves/slurm_logs/viewjob_%a_%A.err
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
ARGS=$(sed -n "$SLURM_ARRAY_TASK_ID"p /scratch/cg3306/climate/CM2P6Param/jobs/viewjob.txt)
module purge
singularity exec --nv --overlay .ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\
	source src.sh;\
	python3 run/view.py $ARGS;\
	"