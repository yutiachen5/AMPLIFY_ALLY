import os
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from collections import defaultdict

import torch
from accelerate import Accelerator

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
        accelerator: Optional[Accelerator] = None,
    ) -> torch.Tensor:
        """Extract sequence-level embeddings for every sample in dataloader.

        In multi-GPU mode, each process handles its data shard. Embeddings are
        gathered from all processes and sorted by global_id so the returned tensor
        is always in dataset order (index i → row i).
        """
        pbar = tqdm(
            desc="Extracting embeddings",
            unit="batch",
            total=len(dataloader),
            disable=(accelerator is not None and not accelerator.is_main_process),
        )
        model.eval()

        local_ids = []
        local_embs = []

        with torch.no_grad():
            for global_id, x, y, pad_mask in dataloader:
                x = x.to(self.device)
                pad_mask = pad_mask.to(self.device)

                emb = model(x, pad_mask, output_hidden_states=True).hidden_states[-1]  # [B, L, D]
                pooled_emb = self._pooling(emb=emb, pad_mask=pad_mask)

                local_ids.append(global_id.cpu())
                local_embs.append(pooled_emb.cpu())
                pbar.update(1)

        model.train()
        pbar.close()

        all_ids = torch.cat(local_ids, dim=0)    # [N_local]
        all_embs = torch.cat(local_embs, dim=0)  # [N_local, D]

        if accelerator is not None and accelerator.num_processes > 1: # dedup batches when number of batches divides evenly across ranks
            all_ids = accelerator.gather(all_ids.to(self.device)).cpu()
            all_embs = accelerator.gather(all_embs.to(self.device)).cpu()

            sort_idx = torch.argsort(all_ids, stable=True)
            # keep only the first occurrence of each id
            seen = set()
            keep = []
            for i in sort_idx.tolist():
                gid = all_ids[i].item()
                if gid not in seen:
                    seen.add(gid)
                    keep.append(i)
            keep = torch.tensor(keep, dtype=torch.long)
            all_embs = all_embs[keep]

        return all_embs.to(dtype=self.dtype)

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
