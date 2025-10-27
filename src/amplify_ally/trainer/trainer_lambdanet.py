import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from copy import deepcopy
from accelerate import Accelerator
from sklearn.model_selection import train_test_split

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

        flag = np.array(flag)
        idx = np.array(idx)

        self.trained_idx = idx[flag[idx] >= 1]
        self.trained_emb = embeddings[flag[idx] >= 1].to(device=device, dtype=dtype)
        self.trained_lambdas = lambdas[flag[idx] >= 1].to(device=device, dtype=dtype)

        self.untrained_idx = idx[flag[idx] < 1]
        self.untrained_emb = embeddings[flag[idx] < 1].to(device=device, dtype=dtype)

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

    def predict(self) -> tuple[list[int], list[float]]:
        loader_te = DataLoader(
            EmbDataset(self.untrained_emb), 
            batch_size=self.per_device_batch_size,
            shuffle=False,
            drop_last=False
        )

        self.model.eval()
        lambda_pred = []

        with torch.no_grad():
            for x in loader_te:
                out = self.model(x)
                lambda_pred += out.squeeze().cpu().tolist()

        self.lambda_pred = np.array(lambda_pred)
        
        return self.lambda_pred

    def reconstruct_lambdas(self, pred_lambdas: list[float]) -> np.ndarray:
        full_lambdas = np.zeros(len(self.trained_idx) + len(self.untrained_idx), dtype=float)

        # Use true lambda to fill trained and pred lambda to fill untrained samples
        full_lambdas[self.trained_idx] = self.trained_lambdas.float().cpu().numpy()
        full_lambdas[self.untrained_idx] = self.lambda_pred

        return torch.tensor(full_lambdas)

    def get_lambdas(
        self,
        val_size: float = 0.2,
        seed: int = 42,
        max_epochs: int = 100,
        print_every: int = 10,
        **kwargs,
    ) -> torch.nn.Module:
        X_train, X_val, y_train, y_val = train_test_split(
            self.trained_emb, self.trained_lambdas, test_size=val_size, random_state=seed
        )

        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        y_val = scaler.transform(y_val.reshape(-1, 1))
        y_train = torch.tensor(y_train, dtype=self.dtype)
        y_val = torch.tensor(y_val, dtype=self.dtype)

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

        best_model = None
        best_val_loss = float("inf")

        for epoch in range(max_epochs):
            train_loss = self.train(loader_tr)
            val_loss = self.validate(loader_val)
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Keep the best model based on val loss
                best_model = deepcopy(self.model)

            if epoch % print_every == 0:
                print(
                    f"[Epoch {epoch:03d}] Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}",
                    flush=True,
                )

        if best_model is None:
            print("Warning: Validation did not improve — keeping the last model.")
            best_model = deepcopy(self.model)
        self.model = best_model

        pred_lambdas = self.predict()
        pred_lambdas = scaler.inverse_transform(pred_lambdas.reshape(-1, 1)).flatten()

        full_lambdas = self.reconstruct_lambdas(pred_lambdas)

        return full_lambdas

