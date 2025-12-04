from tqdm import tqdm
import torch

def get_embedding(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Get the embeddings after each round

    Args:
        model (torch.nn.Module): Model.
        dataloader (torch.utils.data.DataLoader): Dataloader.

    Returns:
       torch.Tensor: embedding of each sample in the training dataloader
    """

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
            emb = model(x, pad_mask, output_hidden_states=True).hidden_states[-1]

            # Mean pooling to get the seq-level representation. 0: valid, inf: false
            pooling_indicator = torch.isfinite(pad_mask).to(torch.float32)
            valid_counts = torch.sum(pooling_indicator, dim=1, keepdim=True)
            pooled_emb = torch.sum(emb*pooling_indicator.unsqueeze(-1), dim=1)/valid_counts # [batch_size, emb_dim], seq-level embeddings
            pooled_emb = pooled_emb.detach().cpu()
            embedding.append(pooled_emb)

            pbar.update(1)
        embedding = torch.cat(embedding, dim=0) # [n_samples, emb_dim], emb is on cpu

    model.train()
    pbar.close()

    return embedding

def pooling(
    emb: torch.Tensor, 
    pad_mask: torch.Tensor,
    pooling: str,
    **kwargs,
) -> torch.Tensor:

    pooling_indicator = torch.isfinite(pad_mask).to(torch.float32)
    valid_counts = torch.sum(pooling_indicator, dim=1, keepdim=True)

    if pooling == "mean":
        pooled_emb = torch.sum(emb*pooling_indicator.unsqueeze(-1), dim=1)/valid_counts # [batch_size, emb_dim], seq-level embeddings

    return pooled_emb.detach().cpu()
