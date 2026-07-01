import os
import numpy as np
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from omegaconf import DictConfig

from ..utils import load_aux_state


@dataclass
class ResumeState:
    lambdas: torch.Tensor
    flag: np.ndarray
    idx_order: np.ndarray
    centroid: np.ndarray | None
    best_reg: dict | None
    dataloader: DataLoader


def restore_from_checkpoint(
    chk_dir: str,
    it: int,
    trainer_cfg: DictConfig,
    n_steps: int,
    accelerator: Accelerator,
    reg: torch.nn.Module,
    optimizer_reg: torch.optim.Optimizer,
    dtype: torch.dtype,
    dataset,
    collator,
    metrics,
) -> ResumeState:
    accelerator.load_state(os.path.join(chk_dir, f"checkpoint_{it}"))

    lambdas, flag, idx_order, centroid, best_reg, optimizer_reg_state = load_aux_state(chk_dir, it, dtype)

    if best_reg is not None:
        reg.load_state_dict({k: v.to(device=accelerator.device, dtype=dtype) for k, v in best_reg.items()})
        accelerator.print(f"[resume] Loaded LambdaNet weights from checkpoint_{it}")

    if optimizer_reg_state is not None:
        optimizer_reg.load_state_dict(optimizer_reg_state)
        accelerator.print(f"[resume] Loaded LambdaNet optimizer state from checkpoint_{it}")

    metrics["num_steps"] = n_steps * trainer_cfg.resume_it
    accelerator.print(f"[resume] resume_it={trainer_cfg.resume_it}: num_steps={metrics['num_steps']}, starting from round {trainer_cfg.resume_it + 1}")

    dataloader = accelerator.prepare_data_loader(DataLoader(
        dataset=dataset.update(idx_order),
        batch_size=trainer_cfg.train.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=trainer_cfg.train.num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=False,
    ))

    return ResumeState(
        lambdas=lambdas,
        flag=flag,
        idx_order=idx_order,
        centroid=centroid,
        best_reg=best_reg,
        dataloader=dataloader,
    )
