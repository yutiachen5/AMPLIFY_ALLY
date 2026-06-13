import torch
from typing import Tuple
from scipy.stats import spearmanr

import os
import numpy as np
import pandas as pd

def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
    vocab_size: int,
) -> Tuple[int, int, int]:
    """Evaluate the model on the dataloader provided.

    Args:
        model (torch.nn.Module): Model.
        dataloader (torch.utils.data.DataLoader): Dataloader.
        loss_fn (torch.nn.modules.loss._Loss): Loss function, returning mean value.
        vocab_size (int): Total number of tokens in the vocabulary.

    Returns:
        Tuple[int,int,int]: Sum of per-token losses, sum of correct predictions, and number of predictions.
    """
    model.eval()
    sum_val_loss, num_val_correct, num_val_pred = 0, 0, 0
    with torch.no_grad():
        for global_id, x, y, pad_mask in dataloader:
            logits = model(x, pad_mask).logits
            val_loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            num_val_pred += torch.sum(y != -100).item()
            sum_val_loss += val_loss.item() * torch.sum(y != -100).item()
            num_val_correct += torch.sum(torch.argmax(logits, dim=-1) == y).item()
    model.train()

    return num_val_pred, sum_val_loss, num_val_correct

def evaluate_proteingym(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader, 
    dataset: torch.utils.data.Dataset, 
    device: torch.device,
    pad_token_id: int,
    dtype: torch.dtype,
):
    """Run ProteinGym evaluation using a pre-built dataloader.
 
    Args:
        model: torch.nn.Module.
        dataloader: torch.utils.data.DataLoader.
        dataset: torch.utils.data.Dataset.
        device: torch.device.
 
    Returns:
        mean Spearman correlation across all assays.
    """
    # reset model scores from any previous evaluation
    for score in dataset.assay_scores.values():
        score["model_scores"][:] = 0.0
 
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            masked_ids  = batch["masked_ids"].to(device) # (B, L), L is max_len
            new_pos = batch["new_pos"] # (B,)
            dms_idx = batch["dms_idx"] # list[str]
            mutants = batch["mutants"] # list[list]
 
            pad_mask = (masked_ids == pad_token_id).to(dtype=dtype)  # (B, L)
            logits = model(masked_ids, pad_mask).logits
            logits = model(masked_ids).logits # (B, L, vocab)
            log_probs = torch.log_softmax(logits, dim=-1) # (B, L, vocab)
 
            for j in range(len(dms_idx)):
                pos_log_probs = log_probs[j, new_pos[j] + 1].cpu() # (vocab,)
                score = dataset.assay_scores[dms_idx[j]]
                for row_idx, wt_id, mt_id in mutants[j]:
                    score["model_scores"][row_idx] += (pos_log_probs[mt_id] - pos_log_probs[wt_id]).item()
 
    # compute per-assay Spearman and return mean
    results = []
    for dms_id, score in dataset.assay_scores.items():
        rho, _ = spearmanr(score["model_scores"], score["dms_scores"])
        results.append(rho)
 
    model.train()
    return float(np.mean(results)) if results else float("nan")