#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J A549-mse
#SBATCH -o /datacommons/igvf-pm/A549/leave-one-out/Dex-200/slurm-mse/outputs/%a.out
#SBATCH -e /datacommons/igvf-pm/A549/leave-one-out/Dex-200/slurm-mse/outputs/%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1-24%23
#SBATCH -p majoroslab-gpu,scavenger-gpu
#SBATCH --nice=100
#SBATCH --mem=50000
#SBATCH --cpus-per-task=1
#
/datacommons/igvf-pm/A549/leave-one-out/Dex-200/slurm-mse/command${SLURM_ARRAY_TASK_ID}.sh