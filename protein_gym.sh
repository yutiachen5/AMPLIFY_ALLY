#!/bin/bash
#SBATCH --job-name=testing
#SBATCH -A naderilab
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=1:00:00
#SBATCH --mem=30G
#SBATCH --array=0-8

MDL_PATH="/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/logs/testing_save_pt_mdl/checkpoints/checkpoint_1/model.pt"
CONFIG_PATH="/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/logs/testing_save_pt_mdl/.hydra/config.yaml"

srun \
	/bin/bash -c "\
	/hpc/group/naderilab/eleanor/env/bin/accelerate launch \
	/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/examples/protein_gym.py --model_path ${MDL_PATH} --config_path ${CONFIG_PATH} --DMS_index ${SLURM_ARRAY_TASK_ID} \
	> /dev/null 2>&1
"