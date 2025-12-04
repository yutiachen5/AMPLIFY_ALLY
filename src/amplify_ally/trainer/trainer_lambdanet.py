import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        idx: np.ndarray,
        embeddings: torch.Tensor,
        lambdas: torch.Tensor,
        flag: np.ndarray,
        accelerator: Accelerator,
        per_device_batch_size_lambdanet: int,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """
        Initialize the LambdaNet trainer.

        Args:
            model (torch.nn.Module): The LambdaNet regression model.
            optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
            idx (np.ndarray): Array of global IDs for each embedding sample.
            embeddings (torch.Tensor): Embeddings aligned with idx.
            lambdas (torch.Tensor): Target lambda values corresponding to embeddings.
            flag (np.ndarray): Indicator array, where `flag[i] >= 1` means the sample (with global ID `i`) was seen/trained by the model, and `< 1` means unseen.
            accelerator (accelerate.Accelerator): Handles device placement and distributed training.
            per_device_batch_size_lambdanet (int): Batch size per device for lambdanet trainer.
            dtype (torch.dtype, optional): Dtype of the pad_mask. Defaults to torch.float32.

        Attributes:
            trained_idx (np.ndarray): Subset of idx where flag >= 1.
            trained_emb (torch.Tensor): Embeddings of trained samples.
            trained_lambdas (torch.Tensor): Lambda values of trained samples.
            untrained_idx (np.ndarray): Subset of idx where flag < 1.
            untrained_emb (torch.Tensor): Embeddings of unseen samples.
        """
        self.model = model
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.per_device_batch_size = per_device_batch_size_lambdanet
        self.scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        self.dtype = dtype
        self.device = device

        # flag = np.array(flag)
        # idx = np.array(idx)

        self.trained_idx = idx[flag[idx] >= 1]
        self.trained_emb = embeddings[flag[idx] >= 1]
        self.trained_lambdas = lambdas[flag[idx] >= 1]

        self.untrained_idx = idx[flag[idx] < 1]
        self.untrained_emb = embeddings[flag[idx] < 1]

        self.trained_lambdas = self.trained_lambdas.to(device=self.device, dtype=self.dtype)
        self.trained_emb = self.trained_emb.to(device=self.device, dtype=self.dtype)
        self.untrained_emb = self.untrained_emb.to(device=self.device, dtype=self.dtype)

        del embeddings

    def train(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for x, y in dataloader:
            # x, y = x.to(device=self.device, dtype=self.dtype), y.to(device=self.device, dtype=self.dtype)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = F.mse_loss(out.squeeze(), y.squeeze())
            # self.accelerator.backward(loss), disable accelerator at this time, single gpu only
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in dataloader:
                # x, y = x.to(device=self.device, dtype=self.dtype), y.to(device=self.device, dtype=self.dtype)
                out = self.model(x)
                loss = F.mse_loss(out.squeeze(), y.squeeze())
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()
        lambda_pred = []

        with torch.no_grad():
            for x in dataloader:
                # x = x.to(device=self.device, dtype=self.dtype)
                out = self.model(x)
                lambda_pred += out.squeeze().cpu().tolist()
        
        return torch.tensor(lambda_pred, dtype=torch.float32, device=self.device) # float32 for scale transformation

    def reconstruct_lambdas(self, pred_lambdas: np.ndarray) -> torch.Tensor:
        full_lambdas = np.zeros(len(self.trained_idx) + len(self.untrained_idx), dtype=float)

        full_lambdas[self.trained_idx] = self.trained_lambdas.detach().to(torch.float32).cpu().numpy()
        full_lambdas[self.untrained_idx] = pred_lambdas

        return torch.tensor(full_lambdas, dtype=torch.float32)

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

        # self.trained_lambdas = self.trained_lambdas.cpu().numpy()
        # X_train, X_val, y_train, y_val = train_test_split(
        #     self.trained_emb, self.trained_lambdas, test_size=val_size, random_state=seed
        # )

        # min-max scaler
        y_min, y_max = y_train.min(), y_train.max()
        scale = (y_max - y_min).clamp_min(1e-12)

        y_train = ((y_train - y_min)/scale).view(-1, 1)
        y_val = ((y_val - y_min)/scale).view(-1, 1)

        # scaler = MinMaxScaler()
        # y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        # y_val = scaler.transform(y_val.reshape(-1, 1))
        # y_train = torch.tensor(y_train, dtype=self.dtype, device=self.device)
        # y_val = torch.tensor(y_val, dtype=self.dtype,  device=self.device)

        loader_tr = DataLoader(
            LambdaSet(X_train, X_val, y_train, y_val, train=True),
            batch_size=self.per_device_batch_size,
            shuffle=True,
            drop_last=True,
        )
        loader_val = DataLoader(
            LambdaSet(X_train, X_val, y_train, y_val, train=False),
            batch_size=self.per_device_batch_size,
            shuffle=True,
            drop_last=True,
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
        pred_lambdas = (pred_lambdas * scale + y_min).detach().cpu().numpy()
        # pred_lambdas = pred_lambdas.detach().cpu().numpy()
        # pred_lambdas = scaler.inverse_transform(pred_lambdas.reshape(-1, 1)).flatten()
        print("Lambda prediction completed.")

        full_lambdas = self.reconstruct_lambdas(pred_lambdas)
        print("Lambda updating completed.")

        return full_lambdas

