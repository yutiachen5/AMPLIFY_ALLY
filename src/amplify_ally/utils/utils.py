import os
import wandb
import numpy as np
from typing import Tuple

import torch


def save_aux_state(
    chk_dir: str,
    it: int,
    lambdas: torch.Tensor,
    flag: np.ndarray,
    idx_order: np.ndarray,
    best_reg: dict | None = None,
) -> None:
    folder = os.path.join(chk_dir, f"checkpoint_{it}")
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "lambdas.npy"), lambdas.detach().cpu().to(torch.float32).numpy())
    np.save(os.path.join(folder, "flag.npy"), flag)
    np.save(os.path.join(folder, "idx_order.npy"), np.asarray(idx_order, dtype=np.int64))
    if best_reg is not None:
        torch.save(best_reg, os.path.join(folder, "lambdanet.pt"))

def load_aux_state(
    chk_dir: str,
    it: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    folder = os.path.join(chk_dir, f"checkpoint_{it}")
    lambdas_path = os.path.join(folder, "lambdas.npy")
    flag_path = os.path.join(folder, "flag.npy")
    idx_order_path = os.path.join(folder, "idx_order.npy")

    missing = [p for p in [lambdas_path, flag_path, idx_order_path] if not os.path.exists(p)]
    if missing:
        print(f"[resume] WARNING: checkpoint files not found: {missing}. Starting from scratch.")
        return None, None, None

    lambdas = torch.from_numpy(np.load(lambdas_path)).to(dtype)
    flag = np.load(flag_path)
    idx_order = np.load(idx_order_path)
    print(f"[resume] Loaded checkpoint state from {folder}")
    return lambdas, flag, idx_order

def get_wandb_run_id(dir: str, resume: bool, is_main_process: bool) -> str:
    run_id_path = os.path.join(dir, "wandb_run_id.txt")

    if resume and os.path.exists(run_id_path):
        with open(run_id_path) as f:
            run_id = f.read().strip()
        print(f"[wandb] Resuming run {run_id}")
    else:
        run_id = wandb.util.generate_id()
        if is_main_process:
            os.makedirs(dir, exist_ok=True)
            with open(run_id_path, "w") as f:
                f.write(run_id)
        print(f"[wandb] New run {run_id}")

    return run_id