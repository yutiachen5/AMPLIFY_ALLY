#!/bin/bash
#SBATCH --job-name=regLr.3e-5_scaleLrFactor.3_ceiling.1e-3_mean_e.2.2_nclusters.512_seed.100_nsamples.12M_nsteps.8k
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1-00:00:00

#SBATCH --output=%x_output.txt
#SBATCH --error=%x_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=255G
#SBATCH --signal=TERM@60

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

source /hpc/group/naderilab/eleanor/AMPLIFY_ALLY/env/bin/activate

echo "[INFO] nodes=${SLURM_JOB_NUM_NODES} gpus_per_task=${SLURM_GPUS_ON_NODE}"
echo "[INFO] master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"

# crucial: make sure trainer.save_steps == strategy.n_steps
srun \
    --unbuffered \
    --kill-on-bad-exit=1 \
    --nodes=$SLURM_JOB_NUM_NODES \
    --ntasks=$SLURM_JOB_NUM_NODES \
    --ntasks-per-node=1 \
    --gpus-per-task=1 \
    /bin/bash -c "\
    /hpc/group/naderilab/eleanor/AMPLIFY_ALLY/env/bin/accelerate launch \
    --config_file=conf/accelerate_gpu.yaml \
    --machine_rank=$SLURM_NODEID \
    --num_cpu_threads_per_process=$SLURM_CPUS_PER_TASK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_processes=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE)) \
    --num_machines=$SLURM_JOB_NUM_NODES \
    --mixed_precision=bf16 \
    --gradient_clipping=1.0 \
    /hpc/group/naderilab/eleanor/AMPLIFY_ALLY/scripts/pretrain.py \
    hydra.run.dir=/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/logs/"$SLURM_JOB_NAME" \
    wandb.dir=/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/logs/"$SLURM_JOB_NAME" \
    wandb.name=$SLURM_JOB_NAME \
    model=[amplify,120M] \
    optimizer=adamw \
    optimizer.lr=0.001 \
    optimizer.betas=[0.9,0.95] \
    optimizer.weight_decay=0.01 \
    scheduler=cosine_decay \
    scheduler.warmup_steps=0 \
    scheduler.final_step=900000 \
    trainer.dir=/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/logs/"$SLURM_JOB_NAME" \
    trainer.max_steps=32000 \
    trainer.train.per_device_batch_size=256 \
    trainer.validation.per_device_batch_size=512 \
    trainer.gradient_accumulation_steps=2 \
    trainer.save_steps=8000 \
    trainer.eval_steps=100 \
    strategy.n_steps=8000 \
    strategy.slack_lr=0 \
    strategy.n_iters=1 \
    strategy.n_clusters=512 \
    strategy.epsilon=2.2 \
    strategy.swap=True \
    strategy.dual_lr_gamma=0.9 \
    strategy.dual_lr_stepsize=800 \
    strategy.max_epochs=100 \
    strategy.patience=5 \
    strategy.per_device_batch_size_emb=512 \
    strategy.per_device_batch_size_kmeans=512 \
    strategy.per_device_batch_size_lambdanet=512 \
    strategy.has_emb=False \
    strategy.write_to_hard_drive=False \
    strategy.print_every=1 \
    strategy.optimizer_lr=3e-5 \
    strategy.max_rds=4 \
    strategy.save_intermediates=False \
    strategy.pooling_method=mean \
    strategy.scale_lr_factor=3 \
    strategy.resume=True \
    seed=100 \
    dataset=uniref50_0.2
"
