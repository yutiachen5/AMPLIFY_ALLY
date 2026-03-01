import torch

import os
import numpy as np
from tqdm import tqdm

def pooling(
    emb: torch.Tensor,
    pad_mask: torch.Tensor,
    pooling: str,
    **kwargs,
) -> torch.Tensor:
    """
    emb: [batch, seq_len, emb_dim]
    pad_mask: [batch, seq_len] where valid positions have finite values
    """
    pooling_indicator = torch.isfinite(pad_mask).to(torch.float32)  # [B, L]
    valid_counts = pooling_indicator.sum(dim=1, keepdim=True)  # [B, 1]

    if pooling == "mean":
        pooled_emb = (emb * pooling_indicator.unsqueeze(-1)).sum(dim=1) / valid_counts  # [B, D]
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")

    return pooled_emb.detach().cpu()


def get_embedding(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    write_to_hard_drive,
    dtype: torch.dtype = torch.float32,
    pooling_type: str = "mean",
    **kwargs,
) -> torch.Tensor | None:
    """Get pooled (sequence-level) embeddings for each sample in dataloader."""

    pbar = tqdm(
        desc="Extract embeddings",
        unit="batch",
        initial=0,
        total=len(dataloader),
    )

    model.eval()
    embedding = []


    with torch.no_grad():
        for global_id, x, y, pad_mask in dataloader:
            x = x.to(device)
            pad_mask = pad_mask.to(device)

            emb = model(x, pad_mask, output_hidden_states=True).hidden_states[-1]  # [B, L, D]

            pooled_emb = pooling(
                emb=emb,
                pad_mask=pad_mask,
                pooling=pooling_type,
            ) 

            if write_to_hard_drive:
                out_file = os.path.join(save_dir, f"seq_{global_id}.pt")
                torch.save(out_file, pooled_emb)
            else:
                embedding.append(pooled_emb)
            pbar.update(1)

    model.train()
    pbar.close()

    if write_to_hard_drive:
        out_file.close()
        return None
    else:
        embedding = torch.cat(embedding, dim=0).to(dtype=dtype)  # [n_samples, emb_dim] on CPU
        return embedding