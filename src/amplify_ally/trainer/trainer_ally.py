import os
import gc
import re
import sys
import json
import signal
import shutil
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed, broadcast_object_list
from deepspeed.utils import safe_get_full_fp32_param

from ..config import config_schema, ConfigError
from ..model import AMPLIFY, AMPLIFYConfig, LambdaNet
from ..metric import Metrics
from ..loss import get_loss, get_lagrangian, update_dual_variables
from ..dataset import get_mlm_dataloader, update_mlm_dataloader, compute_sample_order, get_emb_dataloader, get_proteingym_dataloader
from ..scheduler import get_scheduler
from ..optimizer import get_optimizer
from ..utils import save_aux_state, load_aux_state, get_wandb_run_id
from .trainer_lambdanet import LambdaNetTrainer
from .evaluation import evaluate, evaluate_proteingym
from .embedder import Embedder


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
    run_id = get_wandb_run_id(dir=cfg.wandb.dir, resume=cfg.trainer.resume, is_main_process=accelerator.is_main_process)
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg.wandb.name, # this should be the same as old job if resuming
                "entity": cfg.wandb.entity,
                "config": OmegaConf.to_container(cfg)
                | {"distributed_type": accelerator.distributed_type}
                | {"mixed_precision": accelerator.mixed_precision},
                "tags": cfg.wandb.tags,
                "dir": cfg.wandb.dir,
                "mode": cfg.wandb.mode,
                "anonymous": "allow",
                "resume": "allow" if cfg.trainer.resume else "never",
                "id": run_id, # the run to resume tracking on wandb
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

    # Initialize embedding model, regression head, optimizer, scheduler, and SWE pooling if specified
    model = AMPLIFY(AMPLIFYConfig(**cfg.model, **cfg.tokenizer))
    reg, best_reg = LambdaNet(input_dim=cfg.model.hidden_size), None
    optimizer = get_optimizer(model, **cfg.optimizer)
    optimizer_reg = get_optimizer(reg, **cfg.strategy)
    scheduler = get_scheduler(optimizer, **cfg.scheduler)

    # Log the number of parameters
    accelerator.log({"model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    # Get the dtype for the pad_mask and class_weights
    dtype_pad_mask, dtype_class_weight, dtype_reg_head = torch.float32, torch.float32, torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask, dtype_reg_head = torch.float16, torch.float16
        if accelerator.distributed_type is DistributedType.DEEPSPEED:
            dtype_class_weight = torch.float16
    elif accelerator.mixed_precision == "bf16": # default
        dtype_pad_mask, dtype_reg_head = torch.bfloat16, torch.bfloat16
        if accelerator.distributed_type is DistributedType.DEEPSPEED:
            dtype_class_weight = torch.bfloat16

    # Train, validation Dataloaders
    train_dataloader = get_mlm_dataloader(
        **cfg.tokenizer,
        **cfg.dataset.train,
        **cfg.trainer.train,
        merge=True,
        return_labels=False,
        dtype=dtype_pad_mask,
        seed=cfg.seed,
    )
    eval_dataloaders = get_mlm_dataloader(
        **cfg.tokenizer,
        **cfg.dataset.validation,
        **cfg.trainer.validation,
        merge=False,
        return_labels=False,
        dtype=dtype_pad_mask,
        seed=cfg.seed,
    )
    dataset = train_dataloader.dataset
    collator = train_dataloader.collate_fn
    emb_dataloader = get_emb_dataloader(dataset, collator, **cfg.strategy)
    pg_dataloader, pg_dataset = get_proteingym_dataloader(
        **cfg.dataset.proteingym,
        **cfg.tokenizer,
        **cfg.trainer.train,
        **cfg.strategy,
    )

    # Constrained learning or not
    constrained = (cfg.strategy.epsilon != 1000)

    # Initialize parameters for constrained learning
    rd_offset = 0
    lambdas = torch.zeros(len(dataset), requires_grad=False, dtype=dtype_pad_mask)
    flag = np.zeros(len(dataset))
    dual_lr = cfg.strategy.dual_lr
    idx_order = np.arange(len(dataset))

    # Initialzie lambdanet trainer
    lambdanet_trainer = LambdaNetTrainer(
        model=reg,
        optimizer=optimizer_reg,
        device=accelerator.device,
        seed=cfg.seed,
        accelerator=accelerator,
        dtype=dtype_reg_head,
        **cfg.strategy,
    )

    # Initialize embedder
    embedder = Embedder(
        device=accelerator.device,
        dtype=dtype_pad_mask,
        hidden_size=cfg.model.hidden_size,
        max_length=cfg.trainer.train.max_length,
        **cfg.strategy,
    )

    # Accelerate
    dataloader = train_dataloader
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)

    reg = reg.to(device=accelerator.device, dtype=dtype_reg_head)
    eval_dataloaders = {k: accelerator.prepare_data_loader(v) for k, v in eval_dataloaders.items()}
    emb_dataloader = accelerator.prepare_data_loader(emb_dataloader)

    # Resume block
    if cfg.trainer.resume and it > 0:
        accelerator.load_state(os.path.join(chk_dir, f"checkpoint_{it}")) # restore the emb mdl
        lambdas, flag, idx_order = load_aux_state(chk_dir, it, dtype_pad_mask)
        rd_path = os.path.join(cfg.trainer.dir, "rd_completed.txt")
        if os.path.exists(rd_path):
            rd_offset = int(open(rd_path).read().strip())
            accelerator.print(f"[resume] Resuming from round {rd_offset + 1}")

        # Rebuild the dataloader using the idx order from previous checkpoint
        train_dataloader = DataLoader(
                dataset=dataset.update(idx_order),
                batch_size=cfg.trainer.train.per_device_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=cfg.trainer.train.num_workers,
                prefetch_factor=2,
                pin_memory=True,
                persistent_workers=False,
            )
        dataloader = accelerator.prepare_data_loader(train_dataloader)
        
    # Get loss functions
    loss_fn = get_loss(accelerator.device, "none", **cfg.tokenizer, **cfg.trainer.train, dtype=dtype_class_weight)
    loss_fn_mean = get_loss(accelerator.device, "mean", **cfg.tokenizer, **cfg.trainer.validation, dtype=dtype_class_weight)

    # Flag-based SIGTERM handler: set a flag and checkpoint at a safe point after the batch,
    # so all ranks participate in the collective save_state together.
    _sigterm_received = False

    def handler(signum, _):
        nonlocal _sigterm_received
        _sigterm_received = True
        print(f"Signal {signum} received on rank {accelerator.process_index}, will checkpoint after current batch")

    signal.signal(signal.SIGTERM, handler)

    # Progress bar
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["num_steps"],
        total=cfg.strategy.n_steps * cfg.strategy.max_rds,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
    )

    for rd in range(rd_offset + 1, cfg.strategy.max_rds + 1):
        accelerator.print(f"#### Round {rd} ####")

        # Rebuild train data loader according to the order of informativeness and diversity except for the last rd
        if rd != 1:
            if constrained:
                # Extract embeddings after the first round and replace the old emb with new one in later rds
                embeddings = embedder.get_embedding(model=model, dataloader=emb_dataloader, accelerator=accelerator)

                # Sync lambdas/slacks/flag across GPUs before single-process work.
                # Each GPU updated non-overlapping indices, so an all-reduce SUM merges them.
                if accelerator.num_processes > 1:
                    lambdas_g = (lambdas if torch.is_tensor(lambdas) else torch.as_tensor(lambdas)).to(accelerator.device)
                    flag_g = torch.as_tensor(flag, dtype=torch.float32).to(accelerator.device)
                    dist.all_reduce(lambdas_g, op=dist.ReduceOp.SUM)
                    dist.all_reduce(flag_g, op=dist.ReduceOp.SUM)
                    lambdas = lambdas_g.cpu()
                    flag = flag_g.cpu().numpy()

                # LambdaNet training runs on main process only (small network, full embeddings needed)
                if accelerator.is_main_process:
                    lambdas, best_reg = lambdanet_trainer.get_lambdas(
                        rd=rd,
                        lambdas=lambdas,
                        flag=flag,
                        embeddings=embeddings,
                        save_dir=cfg.trainer.dir,
                        **cfg.strategy,
                    )
                else:
                    lambdas = None
                    best_reg = None

                # Broadcast lambdas only; best_reg stays on main process for checkpointing
                lambdas_list = [lambdas]
                broadcast_object_list(lambdas_list, from_process=0)
                lambdas = lambdas_list[0]

                # Clustering runs on main process only, then idx_order is broadcast
                if accelerator.is_main_process:
                    idx_order = compute_sample_order(
                        embeddings=embeddings,
                        lambdas=lambdas,
                        seed=cfg.seed,
                        **cfg.strategy,
                    )
                else:
                    idx_order = None

                idx_order_list = [idx_order]
                broadcast_object_list(idx_order_list, from_process=0)
                idx_order = idx_order_list[0]

                idx_order, dataloader = update_mlm_dataloader(
                    dataset=dataset,
                    collator=collator,
                    idx_order=idx_order,
                    **cfg.strategy,
                    **cfg.trainer.train,
                )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Unconstrained: compute order on main process, broadcast
                if accelerator.is_main_process:
                    idx_order = compute_sample_order(
                        embeddings=torch.zeros(len(dataset)),
                        lambdas=lambdas,
                        seed=cfg.seed,
                        **cfg.strategy,
                    )
                else:
                    idx_order = None

                idx_order_list = [idx_order]
                broadcast_object_list(idx_order_list, from_process=0)
                idx_order = idx_order_list[0]

                idx_order, dataloader = update_mlm_dataloader(
                    dataset=dataset,
                    collator=collator,
                    idx_order=idx_order,
                    **cfg.strategy,
                    **cfg.trainer.train,
                )

            dataloader = accelerator.prepare_data_loader(dataloader)

            # reset dual lr to initial value 
            dual_lr = cfg.strategy.dual_lr
            
        for global_id, x, y, pad_mask in dataloader:
            global_id = np.array(global_id.cpu())

            # Increment the number of batches
            metrics["local_num_batches"] += 1

            # Extract the lambda for the current batch
            lambdas_current = lambdas[global_id]

            # Keep recored the number of times each sample was seen by the model
            flag[global_id] += 1

            # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
            # called, and the first call to .backward() outside this context manager will trigger the synchronization (accumulate gradients)
            if metrics["local_num_batches"] % cfg.trainer.gradient_accumulation_steps != 0:
                with accelerator.no_sync(model):

                    out = model(x, pad_mask) 
                    logits = out.logits

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
                    lambdas_updated = update_dual_variables(
                        train_loss_seq=train_loss_seq,
                        lambdas_current=lambdas_current,
                        lr_dual=dual_lr,
                        dtype=dtype_reg_head,
                        **cfg.strategy,
                    )

                    lagrangian, constraint_violations = get_lagrangian(
                        device=accelerator.device,
                        train_loss_seq=train_loss_seq,
                        lambdas_current=lambdas_current,
                        lr_dual=dual_lr,
                        **cfg.strategy
                    )
                    accelerator.backward(lagrangian)

                    lambdas[global_id] = lambdas_updated.detach().cpu()

                    metrics["lambda_mean"] = lambdas[flag >= 1].mean().item()
                    metrics["constraint_violations"] = constraint_violations
            else:
                out = model(x, pad_mask) 
                logits = out.logits

                valid_pos = (y != -100)
                train_loss_token = loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), y.view(-1)) # [batch_size * max_len]
                train_loss_token = train_loss_token.view(logits.shape[0], logits.shape[1]) # [batch_size, max_len]

                train_loss_seq = (train_loss_token * valid_pos).sum(dim=1) / valid_pos.sum(dim=1)
                train_loss_batch = loss_fn_mean(logits.view(-1, cfg.tokenizer.vocab_size), y.view(-1))

                # Log metrics
                pbar.update(1)
                metrics["num_steps"] += 1
                metrics["num_batches_in_epoch"] += 1
                metrics["local_num_samples"] += x.shape[0]
                metrics["local_num_tokens"] += (pad_mask == 0).sum().item()
                metrics["local_num_train_pred"] += torch.sum(y != -100).item()
                metrics["local_sum_train_loss"] += train_loss_batch.item() * torch.sum(y != -100).item()
                metrics["local_num_train_correct"] += torch.sum(torch.argmax(logits, dim=-1) == y).item()

                # Compute gradient and update dual variables
                lambdas_updated = update_dual_variables(
                    train_loss_seq=train_loss_seq,
                    lambdas_current=lambdas_current,
                    lr_dual=dual_lr,
                    dtype=dtype_reg_head,
                    **cfg.strategy,
                )

                lagrangian, constraint_violations = get_lagrangian(
                    device=accelerator.device,
                    train_loss_seq=train_loss_seq,
                    lambdas_current=lambdas_current,
                    lr_dual=dual_lr,
                    **cfg.strategy
                )
                accelerator.backward(lagrangian)

                lambdas[global_id] = lambdas_updated.detach().cpu()

                metrics["lambda_mean"] = lambdas[flag >= 1].mean().item()
                metrics["constraint_violations"] = constraint_violations

                # Evaluate the model
                if metrics["num_steps"] % cfg.trainer.eval_steps == 0:
                    for k, v in eval_dataloaders.items():
                        num_val_pred, sum_val_loss, num_val_correct = evaluate(
                            model=model,
                            dataloader=v,
                            loss_fn=loss_fn_mean,
                            vocab_size=cfg.tokenizer.vocab_size,
                        )
                        metrics[f"local_{k}_sum_val_loss"] = sum_val_loss
                        metrics[f"local_{k}_num_val_correct"] = num_val_correct
                        metrics[f"local_{k}_num_val_pred"] = num_val_pred

                if metrics["num_steps"] % cfg.trainer.pg_eval_steps == 0:
                    if accelerator.is_main_process:
                        proteingym_scc = evaluate_proteingym(
                            model=model,
                            dataloader=pg_dataloader,
                            dataset=pg_dataset,
                            device=accelerator.device,
                            pad_token_id=cfg.tokenizer.pad_token_id,
                            dtype=dtype_pad_mask,
                        )
                        metrics["proteingym_scc"] = proteingym_scc

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
                    metrics["lambdanet_learning_rate"] = lambdanet_trainer.optimizer.param_groups[0]["lr"] 
                    metrics.log(accelerator, os.path.join(cfg.wandb.dir, "wandb", "metrics.json"), model)

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

                # Checkpoint on SIGTERM: all ranks are at a safe point (no collective in flight)
                if _sigterm_received:
                    print(f"Checkpointing on rank {accelerator.process_index} after SIGTERM...")
                    accelerator.save_state()
                    if accelerator.is_main_process:
                        save_aux_state(chk_dir, project_config.iteration - 1, lambdas, flag, idx_order, best_reg)
                    accelerator.wait_for_everyone()
                    print(f"Done on rank {accelerator.process_index}")
                    sys.exit(0)

                # Save emb mdl and aux stuff from the main process
                if metrics["num_steps"] % cfg.strategy.n_steps == 0:
                    accelerator.save_state()
                    if accelerator.is_main_process:
                        save_aux_state(chk_dir, project_config.iteration - 1, lambdas, flag, idx_order, best_reg)
                        with open(os.path.join(cfg.trainer.dir, "rd_completed.txt"), "w") as f:
                            f.write(str(rd))
                    break

        # Log metrics
        metrics["num_epochs"] += 1
        metrics["num_batches_in_epoch"] = 0

    # Save the best models with min val ppl on eval tasks
    summary_dir = os.path.join(cfg.trainer.dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    if any(v is not None for v in metrics.best_mdl_state.values()):
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)

        for eval_set in ["uniprot", "oas", "pdb"]:
            if metrics.best_mdl_state[eval_set] is not None:
                unwrapped.load_state_dict({k: v.to(accelerator.device) for k, v in metrics.best_mdl_state[eval_set].items()})
                accelerator.save_model(unwrapped, os.path.join(summary_dir, eval_set))

    if constrained:
        # Save the final lambda values for future analysis — {raw_seq: lambda}
        idx_np = np.asarray(idx_order, dtype=np.int64)
        lambdas_np = lambdas.detach().cpu().to(torch.float32).numpy()
        lambda_dict = {
            dataset.samples[int(idx)][0]: [dataset.samples[int(idx)][1], float(lambdas_np[int(idx)])]
            for idx in idx_np
        }
        if accelerator.is_main_process:
            with open(os.path.join(summary_dir, "lambdas.json"), "w") as f:
                json.dump(lambda_dict, f, indent=2)


    # Make sure that the wandb tracker finishes correctly and close the progress bar
    pbar.close()
    accelerator.end_training()

