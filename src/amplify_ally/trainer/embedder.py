from tqdm import tqdm

import torch

from ..model import SWE_Pooling


class Embedder:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        pooling_method: str = "mean",
        hidden_size: int = 420,
        max_length: int = 512,
        **kwargs,
    ):
        self.device = device
        self.dtype = dtype
        self.pooling_method = pooling_method

        if pooling_method == "swe":
            self.swe_pooling = SWE_Pooling(d_in=hidden_size, num_slices=hidden_size, num_ref_points=max_length, freeze_swe=True).to(self.device)
        else:
            self.swe_pooling = None

    def get_embedding(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Extract sequence-level embeddings for every sample in dataloader."""

        pbar = tqdm(desc="Extracting embeddings", unit="batch", total=len(dataloader))
        model.eval()

        embedding = []

        with torch.no_grad():
            for _, x, _, pad_mask in dataloader:
                x = x.to(self.device)
                pad_mask = pad_mask.to(self.device)

                emb = model(x, pad_mask, output_hidden_states=True).hidden_states[-1]  # [B, L, D]
                pooled_emb = self._pooling(emb=emb, pad_mask=pad_mask)

                embedding.append(pooled_emb)
                pbar.update(1)

        model.train()
        pbar.close()

        return torch.cat(embedding, dim=0).to(dtype=self.dtype)

    def _pooling(
        self,
        emb: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool [B, L, D] token embeddings to [B, D]."""
        pooling_indicator = torch.isfinite(pad_mask)          # [B, L]
        valid_counts = pooling_indicator.sum(dim=1, keepdim=True)  # [B, 1]

        if self.pooling_method == "mean":
            pooled = (emb * pooling_indicator.unsqueeze(-1)).sum(dim=1) / valid_counts
        elif self.pooling_method == "swe":
            pooled = self.swe_pooling(emb, pooling_indicator)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

        return pooled.to(self.dtype).detach().cpu()
