import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import re
import numpy as np
from copy import deepcopy
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ..dataset import LambdaSet, EmbDataset

class LambdaNetTrainer:
    """
    Trainer class for the LambdaNet model.
    Handles training, validation, and prediction for lambda regression.
    """

    def __init__(
        self,
        rd: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        idx: np.ndarray,
        embeddings: torch.Tensor,
        lambdas: torch.Tensor,
        flag: np.ndarray,
        accelerator: Accelerator,
        save_dir: str,
        per_device_batch_size_lambdanet: int,
        resume: bool,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """
        Initialize the LambdaNet trainer.

        Args:
            rd (int): The round index (1-based).
            model (torch.nn.Module): The LambdaNet regression model.
            optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
            idx (np.ndarray): Array of global IDs for each embedding sample.
            embeddings (torch.Tensor): Embeddings aligned with idx.
            lambdas (torch.Tensor): Target lambda values corresponding to embeddings.
            flag (np.ndarray): Indicator array, where `flag[i] >= 1` means the sample (with global ID `i`) was seen/trained by the model, and `< 1` means unseen.
            accelerator (accelerate.Accelerator): Handles device placement and distributed training.
            save_dir (str): Saving directory for lambdanet if resume is true.
            per_device_batch_size_lambdanet (int): Batch size per device for lambdanet trainer.
            resume (bool): use the model in previous round or not.
            dtype (torch.dtype, optional): Dtype of the pad_mask. Defaults to torch.float32.

        Attributes:
            trained_idx (np.ndarray): Subset of idx where flag >= 1.
            trained_emb (torch.Tensor): Embeddings of trained samples.
            trained_lambdas (torch.Tensor): Lambda values of trained samples.
            untrained_idx (np.ndarray): Subset of idx where flag < 1.
            untrained_emb (torch.Tensor): Embeddings of unseen samples.
        """
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.per_device_batch_size = per_device_batch_size_lambdanet
        self.scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        self.dtype = dtype # keep everything float32 in the regression head
        self.device = device
        self.resume = resume
        self.ckpt_path_save = os.path.join(save_dir, f"Round_{rd}", "reg_ckpt.pt")
        self.ckpt_path_load = os.path.join(save_dir, f"Round_{rd-1}", "reg_ckpt.pt")

        os.makedirs(os.path.dirname(self.ckpt_path_save), exist_ok=True)

        print('rd', rd)

        ckpt_exists = (
            rd is not None
            and rd > 1
            and os.path.isfile(self.ckpt_path_load)
            and os.path.getsize(self.ckpt_path_load) > 0
        )

        if self.resume and ckpt_exists: 
            print("Loading lambdanet checkpoint from ", str(self.ckpt_path_load))
            self.model, self.optimizer = self.load_reg_ckpt(model, optimizer, self.ckpt_path_load)
        else:
            self.model, self.optimizer = model, optimizer

        mask = (flag[idx] >= 1)
        self.trained_idx = torch.as_tensor(idx[mask], dtype=torch.long)     # global ids
        self.untrained_idx = torch.as_tensor(idx[~mask], dtype=torch.long)  # global ids
        self.trained_emb = (embeddings[mask]).to(device=self.device) 
        self.untrained_emb = (embeddings[~mask]).to(device=self.device) 
        self.trained_lambdas = lambdas[self.trained_idx].to(self.device) 
        
    def train(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for x, y in dataloader:
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = F.mse_loss(out.squeeze(), y.squeeze())
            self.accelerator.backward(loss)
            total_loss += loss.item()
            self.optimizer.step()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in dataloader:
                out = self.model(x)
                loss = F.mse_loss(out.squeeze(), y.squeeze())
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()
        lambda_pred = []

        with torch.no_grad():
            for x in dataloader:
                out = self.model(x).view(-1)
                lambda_pred.append(out)

        return torch.cat(lambda_pred, dim=0)

    def reconstruct_lambdas(self, pred_lambdas: torch.Tensor) -> torch.Tensor:
        full_lambdas = torch.zeros(self.trained_idx.shape[0] + self.untrained_idx.shape[0], device=self.device, dtype=self.dtype)

        full_lambdas[self.trained_idx] = self.trained_lambdas.detach()
        full_lambdas[self.untrained_idx] = pred_lambdas

        return full_lambdas.detach().cpu()

    def save_reg_ckpt(self, reg, optimizer_reg, ckpt_path):
        if hasattr(self.accelerator, "is_main_process") and not self.accelerator.is_main_process:
            return

        ckpt = {
            "reg_state": {k: v.detach().cpu() for k, v in reg.state_dict().items()},
            "opt_state": optimizer_reg.state_dict() if optimizer_reg is not None else None,
        }

        self.accelerator.save(ckpt, str(ckpt_path))

    def load_reg_ckpt(self, reg, optimizer_reg, ckpt_path, strict=True):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        reg.load_state_dict(ckpt["reg_state"], strict=strict)
        reg.to(device=self.device)

        if optimizer_reg is not None and ckpt["opt_state"] is not None:
            optimizer_reg.load_state_dict(ckpt["opt_state"])

        return reg, optimizer_reg

    def get_lambdas(
        self,
        val_size: float = 0.2,
        seed: int = 42,
        max_epochs: int = 100,
        print_every: int = 10,
        **kwargs,
    ) -> torch.Tensor:
        n = self.trained_emb.shape[0]
        n_val = int(val_size * n)

        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train = self.trained_emb[train_idx]
        X_val   = self.trained_emb[val_idx]
        y_train = self.trained_lambdas[train_idx]
        y_val   = self.trained_lambdas[val_idx]

        # min-max scaler
        y_min, y_max = y_train.min(), y_train.max()
        scale = (y_max - y_min).clamp_min(1e-12)

        y_train = ((y_train - y_min)/scale).view(-1, 1)
        y_val = ((y_val - y_min)/scale).view(-1, 1)

        loader_tr = DataLoader(
            LambdaSet(X_train, X_val, y_train, y_val, train=True),
            batch_size=self.per_device_batch_size,
            shuffle=True,
            drop_last=True,
        )
        loader_val = DataLoader(
            LambdaSet(X_train, X_val, y_train, y_val, train=False),
            batch_size=self.per_device_batch_size,
            shuffle=False,
            drop_last=False,
        )
        loader_te = DataLoader(
            EmbDataset(self.untrained_emb), 
            batch_size=self.per_device_batch_size,
            shuffle=False,
            drop_last=False
        )

        best_state = None
        best_val_loss = float("inf")

        for epoch in range(max_epochs):
            train_loss = self.train(loader_tr)
            val_loss = self.validate(loader_val)
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Keep the best model based on val loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            if epoch % print_every == 0:
                print(
                    f"[Epoch {epoch:03d}] Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}",
                    flush=True,
                )
        print("Lambda training completed.")

        if best_state is None:
            print("Warning: Validation did not improve — keeping the last model.")
        else:
            self.model.load_state_dict(best_state)

        pred_lambdas = self.predict(loader_te)
        pred_lambdas = pred_lambdas * scale + y_min     

        full_lambdas = self.reconstruct_lambdas(pred_lambdas)

        if self.resume:
            self.save_reg_ckpt(self.model, self.optimizer, self.ckpt_path_save)

        return full_lambdas

