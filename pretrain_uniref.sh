#!/bin/bash
#SBATCH --job-name=uniref30M_nsteps.20k_niters.1
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=7-00:00:00

#SBATCH --output=%x_output.txt
#SBATCH --error=%x_error.txt
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!

#SBATCH --cpus-per-gpu=4                # number of cpus per node
#SBATCH --mem=255G                      # memory per node
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end
                                        # will trigger a checkpoint

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Maximum number of threads in the OpenMP parallel region (defaults to 1)
# (called by `torch.distributed.run`, called by `accelerate launch`)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU 

# Activate the virtual environment
source /hpc/group/naderilab/eleanor/AMPLIFY_ALLY/env/bin/activate

echo "[INFO] nodes=${SLURM_JOB_NUM_NODES} gpus_per_task=${SLURM_GPUS_ON_NODE}"
echo "[INFO] master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"

# Run the command on each node
srun \
	--kill-on-bad-exit=1 \
	--nodes=$SLURM_JOB_NUM_NODES \
	--ntasks=$SLURM_JOB_NUM_NODES \
	--cpus-per-gpu=$SLURM_CPUS_PER_GPU \
	--gpus-per-task=$SLURM_GPUS_PER_TASK \
	--ntasks-per-node=1 \
	/bin/bash -c "\
	/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/env/bin/accelerate launch \
	--config_file=conf/accelerate_gpu.yaml \
	--machine_rank=$SLURM_NODEID \
	--num_cpu_threads_per_process=$SLURM_CPUS_PER_GPU \
	--main_process_ip=$MASTER_ADDR \
	--main_process_port=$MASTER_PORT \
	--num_processes=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE)) \
	--num_machines=$SLURM_JOB_NUM_NODES \
	--mixed_precision=bf16 \
	--gradient_clipping=1.0 \
	/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/scripts/pretrain.py \
	hydra.run.dir=logs/$SLURM_JOB_NAME \
	wandb.dir=logs/$SLURM_JOB_NAME \
	wandb.name=$SLURM_JOB_NAME \
	model=[amplify,120M] \
	optimizer=adamw \
	optimizer.lr=0.001 \
	optimizer.betas=[0.9,0.95] \
	optimizer.weight_decay=0.01 \
	scheduler=cosine_decay \
	scheduler.warmup_steps=0 \
	scheduler.final_step=20000 \
	trainer.dir=logs/$SLURM_JOB_NAME \
	trainer.max_steps=2000 \
	trainer.train.per_device_batch_size=256 \
	trainer.validation.per_device_batch_size=512 \
	trainer.gradient_accumulation_steps=2 \
	trainer.save_steps=20000 \
	trainer.eval_steps=100 \
	strategy.n_steps=20000 \
	strategy.slack_lr=0 \
	strategy.n_iters=1 \
	strategy.n_clusters=2048 \
	strategy.epsilon=1.9 \
	strategy.swap=True \
	strategy.dual_lr_gamma=0.9 \
	strategy.dual_lr_stepsize=1000 \
	strategy.max_epochs=200 \
	strategy.patience=20 \
	strategy.per_device_batch_size_emb=512 \
	strategy.per_device_batch_size_kmeans=512 \
	strategy.per_device_batch_size_lambdanet=512 \
    strategy.has_emb=False \
	strategy.write_to_hard_drive=False \
    strategy.print_every=1 \
	strategy.optimizer_lr=1e-4 \
	strategy.resume=True \
    seed=100 \
    dataset=uniref50
"