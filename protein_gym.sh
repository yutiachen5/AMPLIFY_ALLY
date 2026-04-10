#!/bin/bash
#SBATCH --job-name=protein_gym
#SBATCH -A naderilab
#SBATCH -p scavenger-gpu,gpu-common
#SBATCH --gres=gpu:1
#SBATCH --output=%x_output.txt
#SBATCH --error=%x_error.txt
#SBATCH --time=1:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-2

MDL_PATH="/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/logs/uniref30M_nsteps.20k_niters.1_model.8M/checkpoints/checkpoint_5/model.pt"
CONFIG_PATH="/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/logs/uniref30M_nsteps.20k_niters.1_model.8M/.hydra/config.yaml"
OUT_PATH="/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/ProteinGym/output/uniref30M_nsteps.20k_niters.1_model.8M"

srun /bin/bash -c "
    export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
    /hpc/group/naderilab/eleanor/AMPLIFY_ALLY/env/bin/accelerate launch \
        --num_processes=1 \
        --num_machines=1 \
        --mixed_precision=no \
        --dynamo_backend=no \
        /hpc/group/naderilab/eleanor/AMPLIFY_ALLY/examples/protein_gym.py \
        --model_path ${MDL_PATH} \
        --config_path ${CONFIG_PATH} \
        --DMS_index ${SLURM_ARRAY_TASK_ID} \
        --output_scores_folder ${OUT_PATH}
"