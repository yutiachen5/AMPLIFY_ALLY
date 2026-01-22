import os
import gc
import re
import sys
import pytz
import torch
import signal
import shutil
import datetime 
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List
from omegaconf import OmegaConf, DictConfig

from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from deepspeed.utils import safe_get_full_fp32_param

from ..config import config_schema, ConfigError
from ..model import AMPLIFY, AMPLIFYConfig, LambdaNet
from ..metric import Metrics
from ..loss import get_loss, get_lagrangian, update_dual_variables
from ..dataset import get_dataloader, update_dataloader, emb_dataloader
from ..scheduler import get_scheduler
from ..optimizer import get_optimizer
from ..inference import get_embedding, pooling
from .trainer_lambdanet import LambdaNetTrainer


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
    vocab_size: int,
) -> Tuple[int, int, int]:
    """Evaluate the model on the dataloader provided.

    Args:
        model (torch.nn.Module): Model.
        dataloader (torch.utils.data.DataLoader): Dataloader.
        loss_fn (torch.nn.modules.loss._Loss): Loss function, returning mean value.
        vocab_size (int): Total number of tokens in the vocabulary.

    Returns:
        Tuple[int,int,int]: Sum of per-token losses, sum of correct predictions, and number of predictions.
    """
    model.eval()
    sum_val_loss, num_val_correct, num_val_pred = 0, 0, 0
    with torch.no_grad():
        for global_id, x, y, pad_mask in dataloader:
            logits = model(x, pad_mask).logits
            val_loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            num_val_pred += torch.sum(y != -100).item()
            sum_val_loss += val_loss.item() * torch.sum(y != -100).item()
            num_val_correct += torch.sum(torch.argmax(logits, dim=-1) == y).item()
    model.train()

    return num_val_pred, sum_val_loss, num_val_correct

def trainer_ally(cfg: DictConfig) -> None:
    """Entrypoint for training a model with the given configuraiton.

    Args:
        cfg (DictConfig): Hydra configuration.
    """
    config_check = config_schema.validate(cfg)
    if not config_check.is_ok():
        raise ConfigError(config_check)
    it = 0


    chk_dir = os.path.join(cfg.trainer.dir, "checkpoints")

    # Delete the folder if resume is disable and folder exists
    if cfg.trainer.resume is False:
        shutil.rmtree(chk_dir, ignore_errors=True)
    elif os.path.exists(chk_dir):
        # This regular expression was taken from accelerator.load_state()
        it = max(int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0]) for folder in os.listdir(chk_dir))
        # Remove empty checkpoint folders
        while len(os.listdir(os.path.join(chk_dir, f"checkpoint_{it}"))) == 0:
            shutil.rmtree(os.path.join(chk_dir, f"checkpoint_{it}"), ignore_errors=True)
            it -= 1

    # Accelerator object
    project_config = ProjectConfiguration(
        cfg.trainer.dir,
        automatic_checkpoint_naming=True,
        total_limit=cfg.trainer.max_checkpoints,
        iteration=it + 1,
    )
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb",
        project_config=project_config,
    )

    # Initialise the wandb run and pass wandb parameters
    os.makedirs(cfg.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg.wandb.name,
                "entity": cfg.wandb.entity,
                "config": OmegaConf.to_container(cfg)
                | {"distributed_type": accelerator.distributed_type}
                | {"mixed_precision": accelerator.mixed_precision},
                "tags": cfg.wandb.tags,
                "dir": cfg.wandb.dir,
                "mode": cfg.wandb.mode,
                "anonymous": "allow",
                "resume": cfg.trainer.resume,
            }
        },
    )

    # Set the seed
    set_seed(cfg.seed)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.trainer.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.trainer.tf32)

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    # Embedding model, regression head, optimizer, and learning rate scheduler
    model = AMPLIFY(AMPLIFYConfig(**cfg.model, **cfg.tokenizer))
    reg = LambdaNet(input_dim=cfg.model.hidden_size)
    optimizer = get_optimizer(model, **cfg.optimizer)
    optimizer_reg = get_optimizer(reg, **cfg.strategy)
    scheduler = get_scheduler(optimizer, **cfg.scheduler)
    # scheduler_reg = optim.lr_scheduler.StepLR(optimizer_reg, step_size=1, gamma=0.95)

    # Log the number of parameters
    accelerator.log({"model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    # Get the dtype for the pad_mask and class_weights
    # default values for now: dtype_pad_mask: bf16, dtype_class_weight: torch.float32/bf16??, dtype_reg_head: torch.float32
    dtype_pad_mask, dtype_class_weight, dtype_reg_head = torch.float32, torch.float32, torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
        if accelerator.distributed_type is DistributedType.DEEPSPEED:
            dtype_class_weight = torch.float16
    elif accelerator.mixed_precision == "bf16": # default
        dtype_pad_mask = torch.bfloat16
        if accelerator.distributed_type is DistributedType.DEEPSPEED:
            dtype_class_weight = torch.bfloat16

    # Train, validation Dataloaders
    train_dataloader = get_dataloader(
        **cfg.tokenizer,
        **cfg.dataset.train,
        **cfg.trainer.train,
        merge=True,
        return_labels=False,
        dtype=dtype_pad_mask,
    )
    eval_dataloaders = get_dataloader(
        **cfg.tokenizer,
        **cfg.dataset.validation,
        **cfg.trainer.validation,
        merge=False,
        return_labels=False,
        dtype=dtype_pad_mask,
    )
    collator = train_dataloader.collate_fn

    # Initialize parameters for constrained learning
    dataset = train_dataloader.dataset
    lambdas = torch.zeros(len(dataset), requires_grad=False) 
    slacks = torch.zeros(len(dataset), requires_grad=False)
    flag = np.zeros(len(dataset))
    idx_order = np.arange(len(dataset))

    # Accelerate
    dataloader = train_dataloader
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)
    reg = reg.to(device=accelerator.device, dtype=dtype_reg_head)
    eval_dataloaders = {k: accelerator.prepare(v) for k, v in eval_dataloaders.items()}

    # Get loss functions
    loss_fn = get_loss(accelerator.device, "none", **cfg.tokenizer, **cfg.trainer.train, dtype=dtype_class_weight)
    loss_fn_mean = get_loss(accelerator.device, "mean", **cfg.tokenizer, **cfg.trainer.validation, dtype=dtype_class_weight)

    # Save the model when receiving the signal SIGTERM
    def handler(signum, frame):
        print(f"Signal {signum} received on rank {accelerator.process_index}, checkpointing...")
        accelerator.save_state()
        accelerator.wait_for_everyone()
        print(f"Done on rank {accelerator.process_index}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)

    # Progress bar
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["num_steps"],
        total=cfg.trainer.max_steps,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
    )

    # Generate emb for the whole training set using the initial model
    # embeddings = get_embedding(
    #     model, 
    #     accelerator.prepare(emb_dataloader(dataset, collator, **cfg.strategy, **cfg.trainer.train)), 
    #     accelerator.device,
    #     dtype_reg_head,
    # )

    dual_lr = cfg.strategy.dual_lr

    for rd in range(1, cfg.strategy.max_rds + 1): # fixed number of rounds
        print(f"#### Round {rd} ####")

        if rd > 1:
            # Rebuild train data loader according to the order of informativeness and diversity
            embeddings = get_embedding(
                model, 
                accelerator.prepare(emb_dataloader(dataset, collator, **cfg.strategy, **cfg.trainer.train)), 
                accelerator.device,
                dtype_reg_head
            )

            if cfg.strategy.epsilon != 1000:
                print("Lmabdanet training for constrained learning")
                lambdanet_trainer = LambdaNetTrainer(
                    rd=rd - 1, # -1 since it's based on the prev rd
                    model=reg, 
                    optimizer=optimizer_reg, 
                    device=accelerator.device, 
                    idx=idx_order, 
                    embeddings=embeddings, 
                    lambdas=lambdas, 
                    flag=flag, 
                    accelerator=accelerator, 
                    save_dir=cfg.trainer.dir, 
                    dtype=dtype_reg_head,
                    **cfg.strategy
                )

                # Update lambda value for the next rd
                lambdas_tmp = lambdas.detach().clone() if torch.is_tensor(lambdas) else np.array(lambdas, copy=True) # actual
                lambdas = lambdanet_trainer.get_lambdas(**cfg.strategy) # pred

            # Update dataloder based on actual and predicted lambda
            idx_order, dataloader = update_dataloader(
                dataset=dataset,
                collator=collator,
                embeddings=embeddings, 
                idx_order=idx_order, 
                lambdas=lambdas, 
                **cfg.strategy, 
                **cfg.trainer.train,
            )

            # Saving embeddings, lambdas, and regression head checkpoint (only for constrained learning)
            if cfg.strategy.epsilon != 1000:
                np.save(os.path.join(cfg.trainer.dir, f"Round_{rd-1}", "embeddings.npy"), embeddings.numpy())

                idx_np = np.asarray(idx_order, dtype=np.int64)
                flag_np = np.asarray(flag)
                actual_np = lambdas_tmp.detach().cpu().numpy() if torch.is_tensor(lambdas_tmp) else np.asarray(lambdas_tmp)
                pred_np    = lambdas.detach().cpu().numpy() if torch.is_tensor(lambdas) else np.asarray(lambdas)
                df = pd.DataFrame({
                    "idx": idx_np,
                    "flag": flag_np[idx_np],
                    "lambda_act": actual_np[idx_np],
                    "lambda_pred": pred_np[idx_np],
                })
                df.to_csv(os.path.join(cfg.trainer.dir, f"Round_{rd-1}", "lambdas.csv"), index=False, float_format="%.8f")

            dataloader = accelerator.prepare_data_loader(dataloader)
            print("Dataloader updated.")

            del embeddings
            del lambdanet_trainer
            del df, idx_np, flag_np, actual_np, pred_np
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Cache deleted.")

        for iteration in range(cfg.strategy.n_iters): # go through the loader n_iter times before updating the dataloader
            print("Iteration: ", iteration + 1)
            for global_id, x, y, pad_mask in dataloader:
                # Keep the indices of traning samples
                global_id = np.array(global_id.cpu())

                # Increment the number of batches
                metrics["local_num_batches"] += 1

                # Extract the lambda for the current batch
                lambdas_current = lambdas[global_id]
                slacks_current = slacks[global_id]

                # Keep recored the number of times each sample was seen by the model
                flag[global_id] += 1

                # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
                # called, and the first call to .backward() outside this context manager will trigger the synchronization (accumulate gradients)
                if metrics["local_num_batches"] % cfg.trainer.gradient_accumulation_steps != 0:
                    with accelerator.no_sync(model):
                        # Forward pass
                        out = model(x, pad_mask) # output_hidden_states=True if use the statement below
                        logits = out.logits

                        # if iteration == cfg.strategy.n_iters - 1: # replace the emb with the actual emb from the model during the last iter
                        #     embeddings[global_id] = pooling(out.hidden_states[-1], pad_mask, **cfg.strategy)

                        valid_pos = (y != -100) # Only compute the loss on the masked tokens (-100 is for unmasked)
                        train_loss_token = loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), y.view(-1)) # [batch_size * max_len]
                        train_loss_token = train_loss_token.view(logits.shape[0], logits.shape[1]) # [batch_size, max_len]

                        train_loss_seq = (train_loss_token * valid_pos).sum(dim=1) / valid_pos.sum(dim=1)
                        train_loss_batch = loss_fn_mean(logits.view(-1, cfg.tokenizer.vocab_size), y.view(-1))

                        # Log metrics
                        metrics["num_batches_in_epoch"] += 1
                        metrics["local_num_samples"] += x.shape[0]
                        metrics["local_num_tokens"] += (pad_mask == 0).sum().item()
                        metrics["local_num_train_pred"] += torch.sum(y != -100).item()
                        metrics["local_sum_train_loss"] += train_loss_batch.item() * torch.sum(y != -100).item()
                        metrics["local_num_train_correct"] += torch.sum(torch.argmax(logits, dim=-1) == y).item()

                        # Compute gradient and update dual variables
                        if cfg.strategy.swap:
                            lambdas_updated, slacks_updated = update_dual_variables(
                                train_loss_seq=train_loss_seq,
                                lambdas_current=lambdas_current,
                                slacks_current=slacks_current,
                                lr_dual=dual_lr,
                                dtype=dtype_reg_head,
                                **cfg.strategy, 
                            )

                            lagrangian, constraint_violations = get_lagrangian(
                                device=accelerator.device, 
                                train_loss_seq=train_loss_seq, 
                                lambdas_current=lambdas_current, 
                                slacks_current=slacks_current, 
                                lr_dual=dual_lr, 
                                **cfg.strategy
                            )
                            accelerator.backward(lagrangian)
                        else:
                            lagrangian, constraint_violations = get_lagrangian(
                                device=accelerator.device, 
                                train_loss_seq=train_loss_seq, 
                                lambdas_current=lambdas_current, 
                                slacks_current=slacks_current, 
                                lr_dual=dual_lr, 
                                **cfg.strategy
                            )
                            accelerator.backward(lagrangian)

                            lambdas_updated, slacks_updated = update_dual_variables(
                                train_loss_seq=train_loss_seq,
                                lambdas_current=lambdas_current,
                                slacks_current=slacks_current,
                                lr_dual=dual_lr,
                                dtype=dtype_reg_head,
                                **cfg.strategy, 
                            )

                        lambdas[global_id] = lambdas_updated.detach().cpu()
                        slacks[global_id] = slacks_updated.detach().cpu() 

                        metrics["lambda_mean"] = lambdas[flag >= 1].mean().item() # log the mean of ALL lambdas with non-zero flags
                        metrics["slack_mean"] = slacks[flag >= 1].mean().item()
                        metrics["constraint_violations"] = constraint_violations

                else:
                    # Forward pass
                    out = model(x, pad_mask) # x, mask: [batch_size, max_len]
                    logits = out.logits

                    # if iteration == cfg.strategy.n_iters - 1: # replace the emb with the actual emb from the model during the last iter
                    #     embeddings[global_id] = pooling(out.hidden_states[-1], pad_mask, **cfg.strategy)

                    valid_pos = (y != -100)
                    train_loss_token = loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), y.view(-1)) # [batch_size * max_len]
                    train_loss_token = train_loss_token.view(logits.shape[0], logits.shape[1]) # [batch_size, max_len]

                    train_loss_seq = (train_loss_token * valid_pos).sum(dim=1) / valid_pos.sum(dim=1)
                    train_loss_batch = loss_fn_mean(logits.view(-1, cfg.tokenizer.vocab_size), y.view(-1))

                    # Log metrics
                    pbar.update(1)
                    metrics["num_steps"] += 1 # number of gradient updates
                    metrics["num_batches_in_epoch"] += 1
                    metrics["local_num_samples"] += x.shape[0]
                    metrics["local_num_tokens"] += (pad_mask == 0).sum().item()
                    metrics["local_num_train_pred"] += torch.sum(y != -100).item()
                    metrics["local_sum_train_loss"] += train_loss_batch.item() * torch.sum(y != -100).item()
                    metrics["local_num_train_correct"] += torch.sum(torch.argmax(logits, dim=-1) == y).item()

                    # Compute gradient and update dual variables
                    if cfg.strategy.swap:
                        lambdas_updated, slacks_updated = update_dual_variables(
                            train_loss_seq=train_loss_seq,
                            lambdas_current=lambdas_current,
                            slacks_current=slacks_current,
                            lr_dual=dual_lr,
                            dtype=dtype_reg_head,
                            **cfg.strategy, 
                        )

                        lagrangian, constraint_violations = get_lagrangian(
                            device=accelerator.device, 
                            train_loss_seq=train_loss_seq, 
                            lambdas_current=lambdas_current, 
                            slacks_current=slacks_current, 
                            lr_dual=dual_lr, 
                            **cfg.strategy
                        )
                        accelerator.backward(lagrangian)
                    else:
                        lagrangian, constraint_violations = get_lagrangian(
                            device=accelerator.device, 
                            train_loss_seq=train_loss_seq, 
                            lambdas_current=lambdas_current, 
                            slacks_current=slacks_current, 
                            lr_dual=dual_lr, 
                            **cfg.strategy
                        )
                        accelerator.backward(lagrangian)
                        
                        lambdas_updated, slacks_updated = update_dual_variables(
                            train_loss_seq=train_loss_seq,
                            lambdas_current=lambdas_current,
                            slacks_current=slacks_current,
                            lr_dual=dual_lr,
                            dtype=dtype_reg_head,
                            **cfg.strategy, 
                        )

                    lambdas[global_id] = lambdas_updated.detach().cpu()
                    slacks[global_id] = slacks_updated.detach().cpu() 

                    metrics["lambda_mean"] = lambdas[flag >= 1].mean().item() # log the mean of ALL lambdas with non-zero flags
                    metrics["slack_mean"] = slacks[flag >= 1].mean().item()
                    metrics["constraint_violations"] = constraint_violations

                    # Evaluate the model
                    if metrics["num_steps"] % cfg.trainer.eval_steps == 0:
                        for k, v in eval_dataloaders.items():
                            num_val_pred, sum_val_loss, num_val_correct = evaluate(
                                model,
                                v,
                                loss_fn_mean,
                                cfg.tokenizer.vocab_size,
                            )
                            metrics[f"local_{k}_sum_val_loss"] = sum_val_loss
                            metrics[f"local_{k}_num_val_correct"] = num_val_correct
                            metrics[f"local_{k}_num_val_pred"] = num_val_pred

                    # Log metrics
                    if metrics["num_steps"] % cfg.wandb.log_interval == 0:
                        # https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.utils.safe_get_full_grad
                        if accelerator.distributed_type is DistributedType.DEEPSPEED:
                            metrics["grad_norm"] = model.get_global_grad_norm()
                            metrics["weight_norm"] = (
                                sum(safe_get_full_fp32_param(p).norm(2).item() ** 2 for p in model.parameters()) ** 0.5
                            )
                        # DDP
                        else:
                            metrics["grad_norm"] = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
                            metrics["weight_norm"] = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
                        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
                        metrics.log(accelerator, os.path.join(cfg.wandb.dir, "wandb", "metrics.json"))

                    # Gradient clipping
                    if cfg.trainer.gradient_clipping is not None and cfg.trainer.gradient_clipping > 0:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)

                    # Update the parameters and the scheduler
                    optimizer.step()
                    scheduler.step()

                    # Adjust the dual learning rate every x steps
                    if metrics["num_steps"] % cfg.strategy.dual_lr_stepsize == 0:
                        dual_lr *= cfg.strategy.dual_lr_gamma

                    # Reset the gradient
                    optimizer.zero_grad()

                    # Save the model from the main process
                    if metrics["num_steps"] % cfg.trainer.save_steps == 0:
                        accelerator.save_state()

                    if metrics["num_steps"] % cfg.strategy.n_steps == 0:
                        break

        # Adjust dual learning rate dynamically in each round
        dual_lr = cfg.strategy.dual_lr

        # Log metrics
        metrics["num_epochs"] += 1
        metrics["num_batches_in_epoch"] = 0

    # Make sure that the wandb tracker finishes correctly and close the progress bar
    pbar.close()
    accelerator.end_training()

