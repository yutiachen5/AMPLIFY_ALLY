import os
import numpy as np
from tqdm import tqdm
from typing import List
from collections import defaultdict

import torch

from ..model import SWE_Pooling


class Embedder:
    """Handles extraction, pooling, saving, and updating of sequence-level embeddings.

    Args:
        save_dir: Directory to save/load shard files.
        shard_size: Number of samples per shard file.
        dtype: Torch dtype for embeddings.
        pooling_method: One of 'mean' or 'swe'.
        write_to_hard_drive: If True, stream embeddings to disk in shards.
    """

    def __init__(
        self,
        save_dir: str,
        device: torch.device,
        shard_size: int = 1_000_000,
        dtype: torch.dtype = torch.float32,
        pooling_method: str = "mean",
        write_to_hard_drive: bool = True,
        hidden_size: int = 420,
        max_length: int = 512,
        **kwargs,
    ):
        self.save_dir = save_dir
        self.device = device
        self.shard_size = shard_size
        self.dtype = dtype
        self.pooling_method = pooling_method
        self.write_to_hard_drive = write_to_hard_drive

        if pooling_method == "swe":
            self.swe_pooling = SWE_Pooling(d_in=hidden_size, num_slices=hidden_size, num_ref_points=max_length, freeze_swe=True).to(self.device)
        else:
            self.swe_pooling = None

    def get_embedding(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> tuple[torch.Tensor | None, dict | None]:
        """Extract sequence-level embeddings for every sample in dataloader."""

        pbar = tqdm(desc="Extracting embeddings", unit="batch", total=len(dataloader))
        model.eval()

        embedding = []
        shard_id = 0
        shard_emb_buf: list[np.ndarray] = []
        shard_ids_buf: list = []

        with torch.no_grad():
            for global_id, x, y, pad_mask in dataloader:
                x = x.to(self.device)
                pad_mask = pad_mask.to(self.device)

                emb = model(x, pad_mask, output_hidden_states=True).hidden_states[-1]  # [B, L, D]
                pooled_emb = self._pooling(emb=emb, pad_mask=pad_mask)
                global_id = global_id.detach().cpu().tolist()

                if self.write_to_hard_drive:
                    shard_emb_buf.append(pooled_emb.to(dtype=torch.float32).numpy())
                    shard_ids_buf.extend(global_id)
                    if len(shard_ids_buf) >= self.shard_size:
                        shard_id = self._flush_shard(shard_id, shard_ids_buf, shard_emb_buf)
                else:
                    embedding.append(pooled_emb)

                pbar.update(1)

        if self.write_to_hard_drive:
            self._flush_shard(shard_id, shard_ids_buf, shard_emb_buf)

        model.train()
        pbar.close()

        if self.write_to_hard_drive:
            return None, self.build_id_to_loc()
        else:
            return torch.cat(embedding, dim=0).to(dtype=self.dtype), None

    def update_embedding(
        self,
        global_ids: list,
        pooled_emb: torch.Tensor,
        id_to_loc: dict,
    ):
        """Write updated embeddings back into the correct shard positions."""
        shard_updates: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for i, gid in enumerate(global_ids):
            shard_id, local_id = id_to_loc[int(gid)]
            shard_updates[shard_id].append((local_id, gid, i))

        pooled_np = pooled_emb.numpy() if isinstance(pooled_emb, torch.Tensor) else pooled_emb

        for shard_id, entries in shard_updates.items():
            emb_path = os.path.join(self.save_dir, f"shard_{shard_id:04d}.npy")
            emb = np.load(emb_path, mmap_mode="r+")
            for local_id, _, i in entries:
                emb[local_id] = pooled_np[i]
            emb.flush()

    def build_id_to_loc(self) -> dict:
        """Build global_id -> (shard_id, local_id) lookup from saved shards."""
        id_to_loc = {}
        shard_id = 0
        while True:
            ids_path = os.path.join(self.save_dir, f"shard_{shard_id:04d}_ids.npy")
            if not os.path.exists(ids_path):
                break
            for local_id, global_id in enumerate(np.load(ids_path)):
                id_to_loc[int(global_id)] = (shard_id, local_id)
            shard_id += 1
        return id_to_loc

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

    def _save_shard(self, shard_id: int, all_ids: list, all_emb: list[np.ndarray]):
        """Concatenate buffers and write one shard pair to disk."""
        emb_array = np.concatenate(all_emb, axis=0)
        ids_array = np.array(all_ids, dtype=np.int64)
        np.save(os.path.join(self.save_dir, f"shard_{shard_id:04d}.npy"), emb_array)
        np.save(os.path.join(self.save_dir, f"shard_{shard_id:04d}_ids.npy"), ids_array)

    def _flush_shard(
        self, shard_id: int, ids_buf: list, emb_buf: list[np.ndarray]
    ) -> int:
        """Flush buffers to disk if non-empty, clear them, return next shard_id."""
        if ids_buf:
            self._save_shard(shard_id, ids_buf, emb_buf)
            shard_id += 1
            ids_buf.clear()
            emb_buf.clear()
        return shard_id