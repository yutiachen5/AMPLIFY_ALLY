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
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from deepspeed.utils import safe_get_full_fp32_param
from scipy.stats import spearmanr

from ..config import config_schema, ConfigError
from ..model import AMPLIFY, AMPLIFYConfig, LambdaNet
from ..metric import Metrics
from ..loss import get_loss, get_lagrangian, update_dual_variables
from ..dataset import get_mlm_dataloader, update_mlm_dataloader, compute_sample_order, get_emb_dataloader, get_proteingym_dataloader
from ..scheduler import get_scheduler
from ..optimizer import get_optimizer
from ..utils import save_aux_state
from .trainer_lambdanet import LambdaNetTrainer
from .evaluation import evaluate, evaluate_proteingym
from .embedder import Embedder
from .resume import restore_from_checkpoint


def trainer_ally(cfg: DictConfig) -> None:
    """Entrypoint for training a model with the given configuraiton.

    Args:
        cfg (DictConfig): Hydra configuration.
    """
    config_check = config_schema.validate(cfg)
    if not config_check.is_ok():
        raise ConfigError(config_check)

    num_sets = len(cfg.dataset.train.paths)
    if cfg.strategy.max_rds != num_sets:
        raise ValueError(
            f"cfg.strategy.max_rds ({cfg.strategy.max_rds}) must equal the number of dataset "
            f"sets in cfg.dataset.train.paths ({num_sets}) — round rd introduces set rd-1."
        )
    it = 0

    chk_dir = os.path.join(cfg.trainer.dir, "checkpoints")

    # Delete the folder if resume is disable and folder exists 
    if cfg.trainer.resume is False:
        shutil.rmtree(chk_dir, ignore_errors=True)
    elif os.path.exists(chk_dir):
        # This regular expression was taken from accelerator.load_state()
        it = max(int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0]) for folder in os.listdir(chk_dir))
        if cfg.trainer.resume_it is not None:
            it = cfg.trainer.resume_it
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

    # Initialise the wandb run
    os.makedirs(cfg.wandb.dir, exist_ok=True)
    wandb_init_kwargs = {
        "name": cfg.wandb.name,
        "entity": cfg.wandb.entity,
        "config": OmegaConf.to_container(cfg)
        | {"distributed_type": accelerator.distributed_type}
        | {"mixed_precision": accelerator.mixed_precision},
        "tags": cfg.wandb.tags,
        "dir": cfg.wandb.dir,
        "mode": cfg.wandb.mode,
    }
    if cfg.trainer.wandb_run_id:
        wandb_init_kwargs["id"] = cfg.trainer.wandb_run_id
        wandb_init_kwargs["resume"] = "allow"
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={"wandb": wandb_init_kwargs},
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
    reg, best_reg = LambdaNet(input_dim=cfg.model.hidden_size, init_method=cfg.strategy.lambdanet_init_method), None
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
    elif accelerator.mixed_precision == "bf16": 
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
        max_rows_base_set=cfg.strategy.n_steps*cfg.trainer.gradient_accumulation_steps*cfg.trainer.train.batch_size, # Number of samples in base set training    
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

    cumulative_ends = np.cumsum(dataset.set_lengths) # cumulative idx where each set ends
    total_len = int(cumulative_ends[-1])
    emb_dataloader = get_emb_dataloader(dataset, collator, **cfg.strategy)
    pg_dataloader, pg_dataset = get_proteingym_dataloader(
        **cfg.dataset.proteingym,
        **cfg.tokenizer,
        **cfg.trainer.train,
        **cfg.strategy,
    )

    # Constrained learning or not
    constrained = (cfg.strategy.epsilon != 1000)

    # Resume checking
    if cfg.trainer.resume and cfg.trainer.resume_it is None:
        raise ValueError("trainer.resume_it must be set when trainer.resume=True")
    rd_offset = cfg.trainer.resume_it if cfg.trainer.resume else 0

    # Initialize parameters and lambdanet trainer for constrained learning
    lambdas = torch.zeros(total_len, requires_grad=False, dtype=dtype_pad_mask)
    flag = np.zeros(total_len)
    dual_lr = cfg.strategy.dual_lr

    lambdanet_trainer = LambdaNetTrainer(
        model=reg,
        optimizer=optimizer_reg,
        device=accelerator.device,
        seed=cfg.seed,
        accelerator=accelerator,
        dtype=dtype_reg_head,
        **cfg.strategy,
    )

    # Initialize the base set data
    idx_order = np.arange(len(dataset.samples))
    dataset.update(idx_order)  # restrict pool to the base set

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
        dataset.ensure_loaded_through(rd_offset)
        resumed_num_steps = rd_offset * cfg.strategy.n_steps * cfg.strategy.n_iter
        rs = restore_from_checkpoint(
            chk_dir=chk_dir, it=it, trainer_cfg=cfg.trainer, num_steps=resumed_num_steps,
            accelerator=accelerator, reg=reg, optimizer_reg=optimizer_reg,
            dtype=dtype_pad_mask, reg_dtype=dtype_reg_head, dataset=dataset, collator=collator, metrics=metrics,
        )
        lambdas, flag, idx_order, best_reg = rs.lambdas, rs.flag, rs.idx_order, rs.best_reg
        dataloader = rs.dataloader
        
    # Get loss functions
    loss_fn = get_loss(accelerator.device, "none", **cfg.tokenizer, **cfg.trainer.train, dtype=dtype_class_weight)
    loss_fn_mean = get_loss(accelerator.device, "mean", **cfg.tokenizer, **cfg.trainer.validation, dtype=dtype_class_weight)

    # Flag-based SIGTERM handler: set a flag and checkpoint at a safe point after the batch, so all ranks participate in the collective save_state together.
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
        total=cfg.strategy.max_rds * cfg.strategy.n_steps * cfg.strategy.n_iter,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
    )

    for rd in range(rd_offset + 1, cfg.strategy.max_rds + 1):
        round_start_steps = metrics["num_steps"]
        accelerator.print(
            f"\n{'=' * 50}\n{f'ROUND {rd}/{cfg.strategy.max_rds}':^50}\n{'=' * 50}"
            f"\nRound {rd} step budget: {cfg.strategy.n_steps}"
        )

        # Rebuild train data loader according to the order of informativeness and diversity except for the last rd
        if rd != 1:
            dataset.ensure_loaded_through(rd)
            cumulative_end = int(cumulative_ends[rd - 1])
            new_start = int(cumulative_ends[rd - 2])
            accelerator.print(
                f"Introducing set '{dataset.set_names[rd - 1]}' "
                f"({dataset.set_lengths[rd - 1]} samples) — cumulative pool size: {cumulative_end}"
            )

            if constrained:
                # Fit LambdaNet on only the immediately-preceding round's pool
                fit_start = int(cumulative_ends[rd - 3]) if rd >= 3 else 0
                local_new_start = new_start - fit_start

                # Extract embeddings and restrict emb_dataloader to exactly previous pool + new pool.
                dataset.update(np.arange(fit_start, cumulative_end))
                raw_ids, raw_embeddings = embedder.get_embedding(model=model, dataloader=emb_dataloader)
                embeddings = torch.zeros(cumulative_end - fit_start, raw_embeddings.shape[-1], dtype=raw_embeddings.dtype)
                embeddings[raw_ids - fit_start] = raw_embeddings

                # Fit LambdaNet on the previous pool's trained samples and predict lambdas for the newly-introduced held-out set.
                lambdas_local, best_reg = lambdanet_trainer.get_lambdas(
                    rd=rd,
                    lambdas=lambdas[fit_start:cumulative_end],
                    flag=flag[fit_start:cumulative_end],
                    embeddings=embeddings,
                    **cfg.strategy,
                )
                lambdas[fit_start:cumulative_end] = lambdas_local

                # Snapshot LambdaNet's predicted lambda for the newly-introduced set
                # (still flag==0, so lambdas[new_start:cumulative_end] is exactly
                # reconstruct_lambdas's predicted, not empirical, value here) so we can
                # later check whether it's actually associated with real difficulty —
                # i.e. the loss each sample gets the first time it's ever trained.
                predicted_lambda_snapshot = lambdas[new_start:cumulative_end].clone()
                first_visit_ids, first_visit_losses = [], []

                # Rank only the newly-introduced held-out samples among themselves
                idx_order_local = compute_sample_order(
                    embeddings=embeddings[local_new_start:],
                    lambdas=lambdas_local[local_new_start:],
                    seed=cfg.seed,
                    rd=rd,
                    **cfg.strategy,
                )
                idx_order = idx_order_local + new_start

                idx_order, dataloader = update_mlm_dataloader(
                    dataset=dataset,
                    collator=collator,
                    idx_order=idx_order,
                    seed=cfg.seed,
                    **cfg.strategy,
                    **cfg.trainer.train,
                )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Unconstrained learning: randomize idx order for the new pool so
                # sequence length carries no signal along dataloader index (the
                # pool's on-disk order is length-sorted).
                pool_size = cumulative_end - new_start
                rng = np.random.default_rng(cfg.seed + rd - 2)
                idx_order_local = rng.permutation(pool_size)
                idx_order = idx_order_local + new_start

                idx_order, dataloader = update_mlm_dataloader(
                    dataset=dataset,
                    collator=collator,
                    idx_order=idx_order,
                    seed=cfg.seed,
                    **cfg.strategy,
                    **cfg.trainer.train,
                )

            dataloader = accelerator.prepare_data_loader(dataloader)

            # reset dual lr to initial value 
            dual_lr = cfg.strategy.dual_lr
            
        for iter_idx in range(cfg.strategy.n_iter):
            accelerator.print(f"---- Iter {iter_idx + 1}/{cfg.strategy.n_iter} ----")
            for global_id, x, y, pad_mask in dataloader:
                global_id = np.array(global_id.cpu())

                # Increment the number of batches
                metrics["local_num_batches"] += 1

                # Extract the lambda for the current batch
                lambdas_current = lambdas[global_id]

                # True for samples about to be trained on for the very first time ever
                # (checked before the flag increment below) — used to test whether
                # LambdaNet's predicted lambda for this round's new set is actually
                # associated with real difficulty (see the round-end Spearman check).
                first_visit_mask = (flag[global_id] == 0) if (constrained and rd != 1) else None

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

                        if first_visit_mask is not None and first_visit_mask.any():
                            first_visit_ids.append(global_id[first_visit_mask])
                            first_visit_losses.append(train_loss_seq.detach().cpu().numpy()[first_visit_mask])

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

                    if first_visit_mask is not None and first_visit_mask.any():
                        first_visit_ids.append(global_id[first_visit_mask])
                        first_visit_losses.append(train_loss_seq.detach().cpu().numpy()[first_visit_mask])

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
                                model=accelerator.unwrap_model(model),
                                dataloader=pg_dataloader,
                                dataset=pg_dataset,
                                device=accelerator.device,
                                pad_token_id=cfg.tokenizer.pad_token_id,
                                dtype=dtype_pad_mask,
                            )
                            metrics["proteingym_scc"] = proteingym_scc
                        accelerator.wait_for_everyone()

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
                            save_aux_state(chk_dir, project_config.iteration - 1, lambdas, flag, idx_order, best_reg, optimizer_reg.state_dict())
                        accelerator.wait_for_everyone()
                        print(f"Done on rank {accelerator.process_index}")
                        sys.exit(0)

                    # Save emb mdl and aux stuff from the main process once the round (all n_iter reps) is finished
                    if metrics["num_steps"] - round_start_steps == cfg.strategy.n_steps * (iter_idx + 1):
                        if iter_idx == cfg.strategy.n_iter - 1:
                            accelerator.save_state()
                            if accelerator.is_main_process:
                                save_aux_state(chk_dir, project_config.iteration - 1, lambdas, flag, idx_order, best_reg, optimizer_reg.state_dict())
                        break

        # Diagnostic: is LambdaNet's predicted lambda for this round's newly-introduced
        # set actually associated with real difficulty? Correlate the predicted lambda
        # (snapshotted before training started, i.e. before it could be contaminated by
        # any empirical update) against the loss each sample got the first time it was
        # ever trained. This is a main-process-only estimate over whatever fraction of
        # the new set landed on this rank's dataloader shard — a subsample, but still
        # informative for a directional Spearman correlation.
        if constrained and rd != 1 and accelerator.is_main_process:
            if not first_visit_ids:
                accelerator.print(f"[Round {rd}] lambda-vs-loss check: no first-visit samples captured — skipping.")
            else:
                fv_ids = np.concatenate(first_visit_ids)
                fv_losses = np.concatenate(first_visit_losses)
                fv_predicted_lambda = predicted_lambda_snapshot[fv_ids - new_start].to(torch.float32).numpy()

                # train_loss_seq can legitimately be NaN when a sample's random masking
                # selects zero tokens (valid_pos.sum(dim=1) == 0 -> 0/0) — same case
                # update_dual_variables guards against. A single NaN poisons spearmanr's
                # result entirely, so drop those rows before correlating.
                finite = np.isfinite(fv_losses) & np.isfinite(fv_predicted_lambda)
                n_dropped = len(fv_losses) - finite.sum()
                fv_losses, fv_predicted_lambda = fv_losses[finite], fv_predicted_lambda[finite]

                if finite.sum() < 2:
                    accelerator.print(
                        f"[Round {rd}] lambda-vs-loss check: only {finite.sum()} finite samples "
                        f"out of {len(finite)} captured (dropped_nan={n_dropped}) — not enough to correlate."
                    )
                else:
                    rho, pval = spearmanr(fv_predicted_lambda, fv_losses)
                    accelerator.print(
                        f"[Round {rd}] predicted-lambda vs first-visit-loss Spearman "
                        f"rho={rho:.4f} (p={pval:.3g}, n={finite.sum()}, dropped_nan={n_dropped})"
                    )
                    accelerator.log({"lambda_vs_loss_spearman": rho, "lambda_vs_loss_n": int(finite.sum())})

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

