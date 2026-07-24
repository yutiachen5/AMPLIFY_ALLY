import torch
from torch.utils.data import DataLoader

from ..tokenizer import ProteinTokenizer
from .datasets import InMemoryProteinDataset, InMemoryEmbDataset, ProteinGymDataset
from .data_collator import DataCollatorMLM, ProteinGymCollator

import gc
import random
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from typing import Callable, Dict
from collections import defaultdict


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_mlm_dataloader(
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
    batch_size: int,
    mask_probability: int = 0,
    span_probability: float = 0.0,
    span_max: int = 0,
    exclude_special_tokens_replacement: bool = False,
    padding: str = "max_length",
    pad_to_multiple_of: int = 8,
    dtype: torch.dtype = torch.float32,
    merge: bool = False,
    seed: int = 42,
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
        batch_size (int): Batch size.
        to the next dataset (interleaving). Defaults to ``None``.
        mask_probability (int, optional): Ratio of tokens that are masked. Defaults to 0.
        span_probability (float, optional): Probability for the span length. Defaults to 0.0.
        span_max (int, optional): Maximum span length. Defaults to 0.
        exclude_special_tokens_replacement (bool, optional): Exclude the special tokens such as <BOS> or <EOS> from the
        replacement. Defaults to False.
        padding (str, optional): Pad the batch to the longest sequence or to max_length. Defaults to "max_length".
        pad_to_multiple_of (int, optional): Pad to a multiple of. Defaults to 8.
        dtype (torch.dtype, optional): Dtype of the pad_mask. Defaults to torch.float32.
        seed (int): Random seed for workers. Defualts to 42.

    Returns:
        torch.utils.data.DataLoader
    """

    g = torch.Generator()
    g.manual_seed(seed)

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
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        return {
            k: DataLoader(
                InMemoryProteinDataset([v]),
                batch_size=batch_size,
                collate_fn=collator,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
                persistent_workers=False,
                worker_init_fn=seed_worker,
                generator=g,
            )
            for k, v in paths.items()
        }

def get_emb_dataloader(
    dataset: torch.utils.data.Dataset, 
    collator: Callable,
    batch_size_emb: int,
    **kwargs,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size_emb,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

def get_lambdanet_dataloaders(
    embeddings: torch.Tensor,
    lambdas: torch.Tensor,
    flag: np.ndarray,
    batch_size: int,
    val_size: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, DataLoader]:
    mask = (flag >= 1)
    trained_emb = embeddings[mask]
    untrained_emb = embeddings[~mask]
    trained_lambdas = lambdas[mask]

    n_val = int(val_size * trained_emb.shape[0])
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(trained_emb.shape[0], generator=g)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train = trained_emb[train_idx]
    X_val = trained_emb[val_idx]
    y_train = trained_lambdas[train_idx]
    y_val = trained_lambdas[val_idx]
    X_test = untrained_emb

    # min-max scaler
    y_min, y_max = y_train.min(), y_train.max()
    scale = (y_max - y_min).clamp_min(1e-12)

    y_train = ((y_train - y_min)/scale).view(-1, 1)
    y_val = ((y_val - y_min)/scale).view(-1, 1)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_ds = InMemoryEmbDataset(X_train, y_train, split="train")
    val_ds   = InMemoryEmbDataset(X_val, y_val, split="val")
    test_ds  = InMemoryEmbDataset(X_test, None, split="test")

    return scale, y_min, {
        "train": DataLoader(train_ds, **loader_kwargs),
        "val":   DataLoader(val_ds, **loader_kwargs),
        "test":  DataLoader(test_ds, **loader_kwargs),
    }

    
def compute_sample_order(
    embeddings: torch.Tensor,
    lambdas: torch.Tensor,
    seed: int = 42,
    rd: int = 0,
    n_clusters: int = 512,
    batch_size_kmeans: int = 1024,
    epsilon: int = 2,
    **kwargs,
) -> np.ndarray:
    """Compute the index ordering for the next round (clustering + lambda sort).
    Must run on a single process only.

    Returns:
        np.ndarray of sample indices in the new training order.
    """
    if epsilon == 1000:
        print("Unconstrained learning - randomize idx order for the next rd")
        # Fold in `rd` so the permutation differs each round instead of repeating cfg.seed.
        rng = np.random.default_rng(seed + rd)
        return rng.permutation(np.arange(len(lambdas)))

    kmeans_mdl = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=batch_size_kmeans,
        n_init='auto',
    )

    X = embeddings.detach().to(torch.float32).cpu().numpy()
    clusters = kmeans_mdl.fit_predict(X)
    del X, embeddings
    gc.collect()

    sorted_triplets = sorted(
        zip(clusters, lambdas.to(torch.float32).numpy(), range(len(clusters))),
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
            if i < len(cluster_to_samples[c]):
                _, idx = cluster_to_samples[c][i]
                updated_idx_order.append(idx)
    updated_idx_order = np.array(updated_idx_order)
    del cluster_to_samples
    gc.collect()

    return updated_idx_order


def update_mlm_dataloader(
    dataset: torch.utils.data.Dataset,
    collator: Callable,
    idx_order: np.ndarray,
    batch_size: int = 1024,
    num_workers: int = 2,
    seed: int = 42,
    **kwargs,
) -> DataLoader:
    """Build a DataLoader from a pre-computed index ordering.

    Returns:
        (idx_order, torch.utils.data.DataLoader)
    """
    g = torch.Generator()
    g.manual_seed(seed)

    return idx_order, DataLoader(
        dataset=dataset.update(idx_order),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

def get_proteingym_dataloader(
    DMS_reference_file_path: str,
    DMS_data_dir: str,
    vocab_path: str,
    pad_token_id: int,
    mask_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    unk_token_id: int,
    other_special_token_ids: list | None,
    max_length: int = 512,
    batch_size: int = 256,
    excluded_indices: list | None = None,
    pad_to_multiple_of: int = 8,
    **kwargs,
):
    """Build the ProteinGym dataloader. Call once and reuse across evaluations.
 
    Args:
        DMS_reference_file_path: Path to DMS_substitutions.csv.
        DMS_data_dir: Path to folder with per-assay CSV files.
        max_length: Maximum sequence length for windowing.
        batch_size: Number of unique positions per forward pass.
        num_workers: DataLoader workers.
        excluded_indices: List of DMS integer indices to skip.
 
    Returns:
        (dataloader, dataset) — keep the dataset reference to read model_scores back.
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
    dataset = ProteinGymDataset(
        DMS_reference_file_path = DMS_reference_file_path,
        DMS_data_dir = DMS_data_dir,
        tokenizer = tokenizer,
        max_length = max_length,
        excluded_indices = excluded_indices,
    )
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,        # must stay False — order matters for score accumulation
        collate_fn = ProteinGymCollator(pad_token_id, pad_to_multiple_of),
        num_workers = 0, # hard-coded since it is on main process only
        pin_memory = False,
    )
    return dataloader, dataset