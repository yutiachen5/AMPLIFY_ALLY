import os
import numpy as np
from accelerate import Accelerator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..dataset import get_lambdanet_dataloaders


class LambdaNetTrainer:
    """
    Trainer class for the LambdaNet model.
    Instantiate once before the pretraining loop; call `get_lambdas()` each round.
    Model state is kept in memory and reused across rounds.

    Both ranks run LambdaNet training independently (same data + same seed →
    deterministic, identical results).  The caller broadcasts rank 0's output as
    canonical so any tiny floating-point divergence is harmless.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        seed: int,
        accelerator: Accelerator,
        per_device_batch_size_lambdanet: int,
        scale_lr_factor: int,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        self.seed = seed
        self.accelerator = accelerator
        self.per_device_batch_size = per_device_batch_size_lambdanet
        self.scale_lr_factor = scale_lr_factor
        self.dtype = dtype # the dtype used in pretraining
        self.device = device

        self.model = model
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        self.base_lr = optimizer.param_groups[0]["lr"]

        self.lambdas = None
        self.flag = None
        self.trained_idx = None
        self.untrained_idx = None
        self.trained_lambdas = None

    def _setup_round(self, rd: int, lambdas: torch.Tensor, flag: np.ndarray) -> None:
        """Update per-round data and double the LR for a warm restart."""
        self.lambdas = lambdas
        self.flag = flag

        mask = flag >= 1
        self.trained_idx = torch.where(torch.as_tensor(mask))[0]
        self.untrained_idx = torch.where(torch.as_tensor(~mask))[0]
        self.trained_lambdas = lambdas[self.trained_idx]

        # adjust learning rate
        max_lr = 1e-3  # make this in config later??
        scaled_lr = min(self.base_lr * (self.scale_lr_factor ** (rd-2)), max_lr) # rd starts from 2
        for pg in self.optimizer.param_groups:
            pg["lr"] = scaled_lr
        self.accelerator.print(f"[Round {rd}] LR = base × {self.scale_lr_factor}^{rd-2} → {scaled_lr:.2e}")

        # reset scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)

    def train(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
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
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = F.mse_loss(out.squeeze(), y.squeeze())
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()
        lambda_pred = []

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                out = self.model(x).view(-1)
                lambda_pred.append(out)

        return torch.cat(lambda_pred, dim=0).detach().cpu()

    def reconstruct_lambdas(self, pred_lambdas: torch.Tensor) -> torch.Tensor:
        n = self.trained_idx.shape[0] + self.untrained_idx.shape[0]
        full_lambdas = torch.zeros(n, dtype=self.dtype)
        full_lambdas[self.trained_idx] = self.trained_lambdas
        full_lambdas[self.untrained_idx] = pred_lambdas

        return full_lambdas

    def get_lambdas(
        self,
        rd: int,
        lambdas: torch.Tensor,
        flag: np.ndarray,
        embeddings: torch.Tensor,
        val_size: float = 0.2,
        seed: int = 42,
        max_epochs: int = 100,
        print_every: int = 10,
        patience: int = 3,
        num_workers: int = 4,
        **kwargs,
    ) -> torch.Tensor:
        """Train LambdaNet on every rank (same data + seed → identical result).
        Caller broadcasts rank 0's output as canonical.
        """

        def reset_weights(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        # Refresh per-round state and scale LR
        self._setup_round(rd=rd, lambdas=lambdas, flag=flag)

        # Build dataloaders for lambdanet
        scale, y_min, loaders = get_lambdanet_dataloaders(
            embeddings=embeddings,
            lambdas=self.lambdas,
            flag=self.flag,
            device=self.device,
            batch_size=self.per_device_batch_size,
            val_size=val_size,
            seed=seed,
            num_workers=num_workers,
            dtype=self.dtype,
        )

        # Training loop with early stopping
        best_state = None
        best_val_loss = float("inf")
        n_no_improve = 0

        for epoch in range(max_epochs):
            train_loss = self.train(loaders["train"])
            val_loss = self.validate(loaders["val"])
            self.scheduler.step(val_loss)

            # track grad and weight norm
            weight_norm = sum(p.data.norm(2).item() ** 2 for p in self.model.parameters()) ** 0.5
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                n_no_improve = 0
            else:
                n_no_improve += 1

            if epoch % print_every == 0:
                self.accelerator.print(
                    f"[Round {rd} | Epoch {epoch:03d}] "
                    f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
                    f"grad_norm: {grad_norm:.4f} | weight_norm: {weight_norm:.4f}",
                )

            if n_no_improve >= patience:
                self.accelerator.print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                break

        if best_state is None:
            self.accelerator.print("Warning: Validation did not improve — keeping the last model.")
        else:
            self.model.load_state_dict(best_state)

        # Predict & reconstruct
        pred_lambdas = self.predict(loaders["test"])
        pred_lambdas = pred_lambdas * scale + y_min

        # Construct lambdas for the next round of pretraining
        full_lambdas = self.reconstruct_lambdas(pred_lambdas)

        # Free per-round data
        self.lambdas = None
        self.flag = None
        self.trained_idx = None
        self.untrained_idx = None
        self.trained_lambdas = None

        return full_lambdas, best_state
