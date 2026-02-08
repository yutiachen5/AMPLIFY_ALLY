import torch
from torch.utils.data import DataLoader

from ..tokenizer import ProteinTokenizer
from .datasets import InMemoryProteinDataset
from .data_collator import DataCollatorMLM

import gc
import math
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from typing import List, Callable
from collections import defaultdict


def get_dataloader(
    vocab_path: str,
    pad_token_id: int,
    mask_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    unk_token_id: int,
    other_special_token_ids: list | None,
    paths: dict,
    max_length: int,
    random_truncate: bool,
    return_labels: bool,
    num_workers: int,
    per_device_batch_size: int,
    samples_before_next_set: list | None = None,
    mask_probability: int = 0,
    span_probability: float = 0.0,
    span_max: int = 0,
    exclude_special_tokens_replacement: bool = False,
    padding: str = "max_length",
    pad_to_multiple_of: int = 8,
    dtype: torch.dtype = torch.float32,
    merge: bool = False,
    **kwargs,
) -> DataLoader:
    """Public wrapper for constructing a ``torch`` dataloader.

    Args:
        vocab_path (str): Path to the vocabulary file to load.
        pad_token_id (int): <PAD> token index in the vocab file.
        mask_token_id (int): <MASK> token index in the vocab file.
        bos_token_id (int): <BOS> token index in the vocab file.
        eos_token_id (int): <EOS> token index in the vocab file.
        unk_token_id (int): <UNK> token index in the vocab file.
        other_special_token_Unknown ids (list | None): List of other special tokens.
        paths (dict): Dict of name:paths to the CSV files to read.
        max_length (int): Maximum sequence length.
        random_truncate (bool): Truncate the sequence to a random subsequence of if longer than truncate.
        return_labels (bool): Return the protein labels.
        num_workers (int): Number of workers for the dataloader.
        per_device_batch_size (int): Batch size for each GPU.
        samples_before_next_set (list | None, optional): Number of samples of each dataset to return before moving
        to the next dataset (interleaving). Defaults to ``None``.
        mask_probability (int, optional): Ratio of tokens that are masked. Defaults to 0.
        span_probability (float, optional): Probability for the span length. Defaults to 0.0.
        span_max (int, optional): Maximum span length. Defaults to 0.
        exclude_special_tokens_replacement (bool, optional): Exclude the special tokens such as <BOS> or <EOS> from the
        replacement. Defaults to False.
        padding (str, optional): Pad the batch to the longest sequence or to max_length. Defaults to "max_length".
        pad_to_multiple_of (int, optional): Pad to a multiple of. Defaults to 8.
        dtype (torch.dtype, optional): Dtype of the pad_mask. Defaults to torch.float32.

    Returns:
        torch.utils.data.DataLoader
    """
    tokenizer = ProteinTokenizer(
        vocab_path,
        pad_token_id,
        mask_token_id,
        bos_token_id,
        eos_token_id,
        unk_token_id,
        other_special_token_ids,
    )
    collator = DataCollatorMLM(
        tokenizer,
        max_length,
        random_truncate,
        return_labels,
        mask_probability,
        span_probability,
        span_max,
        exclude_special_tokens_replacement,
        padding,
        pad_to_multiple_of,
        dtype,
    )

    if merge:
        return DataLoader(
            InMemoryProteinDataset(paths.values()),
            batch_size=per_device_batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=False,
        )
    else:
        return {
            k: DataLoader(
                InMemoryProteinDataset([v]),
                batch_size=per_device_batch_size,
                collate_fn=collator,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
                persistent_workers=False,
            )
            for k, v in paths.items()
        }

def emb_dataloader(
    dataset: torch.utils.data.Dataset, 
    collator: Callable,
    per_device_batch_size_emb: int,
    num_workers: int,
    **kwargs,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=per_device_batch_size_emb,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=False,
    )

def update_dataloader(
    dataset: torch.utils.data.Dataset,
    collator: Callable,
    embeddings: torch.Tensor,
    idx_order: np.array,
    lambdas: torch.Tensor,
    seed: int,
    n_clusters: int,
    per_device_batch_size_kmeans: int,
    per_device_batch_size: int,
    num_workers: int,
    epsilon: int,
    **kwargs,
) -> DataLoader:
    """Update the order of samples in the dataloader according to informativeness and diversity

    Args:
        embeddings (torch.Tensor). Sequence-level representation.
        lambdas (torch.Tensor). Informativeness of each sequence.
        idx_order (List). List of global ids for the training samples.
        n_clusters (int): Number of KMeans clusters. Defaults to 4_000.
        seed (int): Random seed. Defaults to 0.
        per_device_batch_size_kmeans (int): Batch size for each GPU when doing clustering.
        
    Returns:
        torch.utils.data.DataLoader
    """
    # shuffle the index list for unconstrained learning, the lambda values do not matter since they are all zeros.
    if epsilon == 1000:
        print("Unconstrained learning - randomize idx order for the next rd")
        updated_idx_order = np.random.permutation(idx_order)
        return updated_idx_order, DataLoader(
            dataset=dataset.update(updated_idx_order),
            batch_size=per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=False,
        )


    clusters = []
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, 
        random_state=seed, 
        batch_size=per_device_batch_size_kmeans, 
        n_init='auto',
    )

    X = embeddings.detach().to(torch.float32).cpu().numpy()
    clusters = kmeans.fit_predict(X)
    del X, embeddings
    gc.collect()
    
    # lambdas is global-id axis; align it to the embedding/cluster order (which is idx_order)
    lambdas_aligned = lambdas.to(torch.float32).numpy()[idx_order]   # now aligned with idx_order

    sorted_triplets = sorted(
        zip(clusters, lambdas_aligned, idx_order),
        key=lambda t: (t[0], -t[1])   
    )

    sorted_clusters, sorted_lambdas, sorted_idxs = zip(*sorted_triplets)
    del clusters, sorted_triplets
    gc.collect()

    cluster_to_samples = defaultdict(list)
    for c, l, idx in zip(sorted_clusters, sorted_lambdas, sorted_idxs):
        cluster_to_samples[c].append((l, idx))
    del sorted_clusters, sorted_lambdas, sorted_idxs
    gc.collect()

    updated_idx_order = []
    max_len = max(len(v) for v in cluster_to_samples.values())

    for i in range(max_len):                
        for c in range(n_clusters):         
            if i < len(cluster_to_samples[c]):   # skip if cluster shorter
                l, idx = cluster_to_samples[c][i]
                updated_idx_order.append(idx)
    updated_idx_order = np.array(updated_idx_order)
    
    del cluster_to_samples
    gc.collect()
    
    return updated_idx_order, DataLoader(
        dataset=dataset.update(updated_idx_order),
        batch_size=per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=False,
    )

