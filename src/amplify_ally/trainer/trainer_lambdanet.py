import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from accelerate import Accelerator

from ..dataset import get_reg_dataloaders_from_saved_emb_set, get_reg_dataloaders_from_in_memory_emb_set


_VERSION = "2026-05-01"

class LambdaNetTrainer:
    """
    Trainer class for the LambdaNet model.
    Instantiate once before the pretraining loop; call `get_lambdas()` each round.
    Model state is kept in memory and reused across rounds.
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
        self.dtype = dtype
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

    # ------------------------------------------------------------------
    # Round setup
    # ------------------------------------------------------------------

    def _setup_round(self, rd: int, lambdas: torch.Tensor, flag: np.ndarray) -> None:
        """Update per-round data and double the LR for a warm restart."""
        self.lambdas = lambdas
        self.flag = flag

        mask = flag >= 1
        self.trained_idx = torch.where(torch.as_tensor(mask))[0]
        self.untrained_idx = torch.where(torch.as_tensor(~mask))[0]
        self.trained_lambdas = lambdas[self.trained_idx]

        # adjust learning rate
        scaled_lr = self.base_lr * (self.scale_lr_factor ** (rd - 1))
        for pg in self.optimizer.param_groups:
            pg["lr"] = scaled_lr
        print(f"[Round {rd}] LR = base × {self.scale_lr_factor}^{rd-1} → {scaled_lr:.2e}")

        # reset scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)

    # ------------------------------------------------------------------
    # Core train / validate / predict of regression head
    # ------------------------------------------------------------------

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
        emb_dir: str,
        embeddings: torch.Tensor | None = None,
        val_size: float = 0.2,
        seed: int = 42,
        max_epochs: int = 100,
        print_every: int = 10,
        patience: int = 3,
        num_workers: int = 4,
        write_to_hard_drive: bool = True,
        has_emb: bool = False,
        resume: bool = False,
        **kwargs,
    ) -> torch.Tensor:

        def reset_weights(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        # Refresh per-round state and scale LR
        self._setup_round(rd=rd, lambdas=lambdas, flag=flag)
        if not resume: # reset the model weights
            self.model.apply(reset_weights)

        # Build dataloaders
        if write_to_hard_drive:
            if has_emb:
                emb_dir = "/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/logs/sprot_nsteps.100/embeddings"

            loaders = get_reg_dataloaders_from_saved_emb_set(
                emb_dir=emb_dir,
                lambdas=self.lambdas,
                flag=self.flag,
                batch_size=self.per_device_batch_size,
                val_size=val_size,
                seed=seed,
                num_workers=num_workers,
                dtype=self.dtype,
            )
            scale, y_min = None, None
        else:
            scale, y_min, loaders = get_reg_dataloaders_from_in_memory_emb_set(
                embeddings=embeddings,
                lambdas=self.lambdas,
                flag=self.flag,
                device=self.device,
                batch_size=self.per_device_batch_size,
                val_size=val_size,
                seed=seed,
                num_workers=num_workers,
            )

        # Training loop with early stopping
        best_state = None
        best_val_loss = float("inf")
        n_no_improve = 0

        for epoch in range(max_epochs):
            train_loss = self.train(loaders["train"])
            val_loss = self.validate(loaders["val"])
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                n_no_improve = 0
            else:
                n_no_improve += 1

            if epoch % print_every == 0:
                print(
                    f"[Round {rd} | Epoch {epoch:03d}] "
                    f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}",
                    flush=True,
                )

            if n_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                break

        print("Lambda training completed.")

        if best_state is None:
            print("Warning: Validation did not improve — keeping the last model.")
        else:
            self.model.load_state_dict(best_state)

        # Predict & reconstruct
        pred_lambdas = self.predict(loaders["test"])
        if not write_to_hard_drive:
            pred_lambdas = pred_lambdas * scale + y_min

        full_lambdas = self.reconstruct_lambdas(pred_lambdas)

        # Free per-round data
        self.lambdas = None
        self.flag = None
        self.trained_idx = None
        self.untrained_idx = None
        self.trained_lambdas = None

        return full_lambdas