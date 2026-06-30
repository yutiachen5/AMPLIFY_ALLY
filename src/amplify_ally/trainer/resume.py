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
    rd_offset: int
    dataloader: DataLoader
    resume_from_mid_rd: bool


def restore_from_checkpoint(
    chk_dir: str,
    it: int,
    cfg: DictConfig,
    accelerator: Accelerator,
    reg: torch.nn.Module,
    optimizer_reg: torch.optim.Optimizer,
    dtype_pad_mask: torch.dtype,
    dtype_reg_head: torch.dtype,
    dataset,
    collator,
    metrics,
) -> ResumeState:
    accelerator.load_state(os.path.join(chk_dir, f"checkpoint_{it}"))

    lambdas, flag, idx_order, centroid, best_reg, optimizer_reg_state = load_aux_state(chk_dir, it, dtype_pad_mask)

    if best_reg is not None:
        reg.load_state_dict({k: v.to(device=accelerator.device, dtype=dtype_reg_head) for k, v in best_reg.items()})
        accelerator.print(f"[resume] Loaded LambdaNet weights from checkpoint_{it}")

    if optimizer_reg_state is not None:
        optimizer_reg.load_state_dict(optimizer_reg_state)
        accelerator.print(f"[resume] Loaded LambdaNet optimizer state from checkpoint_{it}")

    rd_offset = 0
    rd_path = os.path.join(cfg.trainer.dir, "rd_completed.txt")
    if os.path.exists(rd_path):
        rd_offset = int(open(rd_path).read().strip())
        accelerator.print(f"[resume] Resuming from round {rd_offset + 1}")

    resume_from_mid_rd = metrics["num_steps"] % cfg.strategy.n_steps != 0

    dataloader = accelerator.prepare_data_loader(DataLoader(
        dataset=dataset.update(idx_order),
        batch_size=cfg.trainer.train.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.trainer.train.num_workers,
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
        rd_offset=rd_offset,
        dataloader=dataloader,
        resume_from_mid_rd=resume_from_mid_rd,
    )
