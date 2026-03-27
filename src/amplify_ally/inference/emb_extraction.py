import torch

import os
import numpy as np
from tqdm import tqdm

from typing import List


def update_embedding(
    global_id: list,
    pooled_emb: torch.Tensor,
    emb_save_dir: str,
    shard_size: int = 1_000_000,
    **kwargs,
):
    from collections import defaultdict

    # group by shard so each file is opened once per call
    shard_updates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, gid in enumerate(global_id):
        shard_id  = gid // shard_size
        local_idx = gid % shard_size
        shard_updates[shard_id].append((local_idx, gid, i))

    for shard_id, entries in shard_updates.items():
        emb_path = os.path.join(emb_save_dir, f"shard_{shard_id:04d}.npy")
        ids_path = os.path.join(emb_save_dir, f"shard_{shard_id:04d}_ids.npy")

        ids = np.load(ids_path)
        emb = np.load(emb_path, mmap_mode="r+")

        for local_idx, gid, i in entries:
            assert ids[local_idx] == gid, (
                f"ID mismatch at shard={shard_id} local_idx={local_idx}: "
                f"expected {gid}, got {ids[local_idx]}."
            )
            emb[local_idx] = pooled_emb[i].numpy() if isinstance(pooled_emb, torch.Tensor) else pooled_emb[i]

        emb.flush()  # once per shard

def pooling(
    emb: torch.Tensor,
    pad_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    pooling_method: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """
    emb: [batch, seq_len, emb_dim]
    pad_mask: [batch, seq_len] where valid positions have finite values
    """
    pooling_indicator = torch.isfinite(pad_mask)  # [B, L]
    valid_counts = pooling_indicator.sum(dim=1, keepdim=True)  # [B, 1]

    if pooling_method == "mean":
        pooled_emb = (emb * pooling_indicator.unsqueeze(-1)).sum(dim=1) / valid_counts  # [B, D]
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")

    return pooled_emb.to(dtype).detach().cpu()

    
def save_embedding(
    shard_id: int,
    all_ids: list,
    all_emb: list[np.ndarray],
    emb_save_dir: str,
):
    """Save one shard: a consolidated emb array + companion ids array."""
    emb_array = np.concatenate(all_emb, axis=0)          # [N_shard, D]
    ids_array = np.array(all_ids, dtype=np.int64)         # [N_shard]
 
    emb_path = os.path.join(emb_save_dir, f"shard_{shard_id:04d}.npy")
    ids_path = os.path.join(emb_save_dir, f"shard_{shard_id:04d}_ids.npy")
 
    np.save(emb_path, emb_array)
    np.save(ids_path, ids_array)

def get_embedding(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    write_to_hard_drive,
    dtype: torch.dtype = torch.float32,
    pooling_method: str = "mean",
    has_emb: bool = False,
    shard_size: int = 1_000_000,
    **kwargs,
) -> torch.Tensor | None:
    """Get sequence-level embeddings for each sample in dataloader."""
    if has_emb:
        print("skip emb generation, loading emb from given path")
        return None

    pbar = tqdm(
        desc="Extracting embeddings",
        unit="batch",
        initial=0,
        total=len(dataloader),
    )

    model.eval()
    embedding = []

    # sharding
    shard_id = 0
    shard_emb_buf: list[np.ndarray] = []
    shard_ids_buf: list = []

    def flush_shard():
        nonlocal shard_id
        if shard_ids_buf:
            save_embedding(shard_id, shard_ids_buf, shard_emb_buf, save_dir)
            shard_id += 1
            shard_emb_buf.clear()
            shard_ids_buf.clear()

    with torch.no_grad():
        for global_id, x, y, pad_mask in dataloader:
            x = x.to(device)
            pad_mask = pad_mask.to(device)

            emb = model(x, pad_mask, output_hidden_states=True).hidden_states[-1]  # [B, L, D]
            pooled_emb = pooling(
                emb=emb,
                pad_mask=pad_mask,
                pooling_method=pooling_method,
                dtype=dtype,
            ) 
            global_id = global_id.detach().cpu().tolist()

            if write_to_hard_drive:
                # save_embedding(global_id, pooled_emb, save_dir)
                shard_emb_buf.append(pooled_emb.numpy())
                shard_ids_buf.extend(global_id)
 
                if len(shard_ids_buf) >= shard_size:
                    flush_shard()
            else:
                embedding.append(pooled_emb)
            pbar.update(1)

    # flush any remaining samples into a final partial shard
    if write_to_hard_drive:
        flush_shard()

    model.train()
    pbar.close()

    if write_to_hard_drive:
        return None
    else:
        embedding = torch.cat(embedding, dim=0).to(dtype=dtype)  # [n_samples, emb_dim] on CPU
        return embedding