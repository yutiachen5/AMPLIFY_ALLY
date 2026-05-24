import torch
import torch.nn as nn

import os
import numpy as np
from tqdm import tqdm

from .swe import SWE_Pooling


def pooling(
    emb: torch.Tensor,
    pad_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    pooling_method: str = "mean",
    swe_pooling: nn.Module = None,
    **kwargs,
) -> torch.Tensor:
    """
    emb: [batch, seq_len, emb_dim]
    pad_mask: [batch, seq_len] where valid positions have finite values
    """
    _, max_length, hidden_size = emb.shape

    pooling_indicator = torch.isfinite(pad_mask)  # [B, L], 1=valid, 0=padding
    valid_counts = pooling_indicator.sum(dim=1, keepdim=True)  # [B, 1]

    if pooling_method == "mean":
        pooled_emb = (emb * pooling_indicator.unsqueeze(-1)).sum(dim=1) / valid_counts  # [B, D] in float32
    elif pooling_method == "swe":
        pooled_emb = swe_pooling(emb, pooling_indicator)  # [B, hidden_size]
    else:
        raise ValueError(f"Unsupported pooling: {pooling_method}")

    return pooled_emb.to(dtype)

def save_embedding(global_id: list, pooled_emb: torch.Tensor, emb_save_dir: str):
    for i, gid in enumerate(global_id):
        out_file = os.path.join(emb_save_dir, f"seq_{int(gid)}.npy")
        np.save(out_file, pooled_emb[i])

def get_embedding(
    model: torch.nn.Module,
    swe_pooling: torch.nn.Module | None,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    write_to_hard_drive: bool,
    dtype: torch.dtype = torch.float32,
    pooling_method: str = "mean",
    has_emb: bool = False,
    **kwargs,
) -> tuple[torch.Tensor | None, torch.nn.Module | None]:

    if has_emb and write_to_hard_drive:
        print("skip emb generation, loading emb from given path")
        return None

    model.eval()
    embedding = []

    with torch.no_grad():
        for global_id, x, _, pad_mask in tqdm(dataloader, desc="Extract embeddings", unit="batch"):
            x = x.to(device)
            pad_mask = pad_mask.to(device)
            emb = model(x, pad_mask, output_hidden_states=True).hidden_states[-1]  # [B, L, D]

            pooled_emb = pooling(
                emb=emb,
                pad_mask=pad_mask,
                pooling_method=pooling_method,
                swe_pooling=swe_pooling,
                dtype=dtype,
            ).detach().cpu()  
            global_id = global_id.detach().cpu().tolist()

            if write_to_hard_drive:
                save_embedding(global_id, pooled_emb, save_dir)
            else:
                embedding.append(pooled_emb)

    model.train()

    if write_to_hard_drive:
        return None

    return torch.cat(embedding, dim=0)
