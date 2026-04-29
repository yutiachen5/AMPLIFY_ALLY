import torch
from torch.utils.data import DataLoader, TensorDataset

from ..tokenizer import ProteinTokenizer
from .datasets import InMemoryProteinDataset, SavedEmbDataset, InMemoryEmbDataset
from .data_collator import DataCollatorMLM

import gc
import math
import random
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

from typing import List, Callable, Dict
from collections import defaultdict


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
        seed (int): Random seed for workers. Defualts to 42.

    Returns:
        torch.utils.data.DataLoader
    """

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

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
            batch_size=per_device_batch_size,
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
                batch_size=per_device_batch_size,
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
    per_device_batch_size_emb: int,
    **kwargs,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=per_device_batch_size_emb,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )

def get_reg_dataloaders_from_saved_emb_set(
    emb_dir: str,
    lambdas: torch.Tensor,
    flag: np.ndarray,
    batch_size: int,
    val_size: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    dtype: torch.dtype = torch.float32,
    kmeans: bool = False,
) -> Dict[str, DataLoader]:
    """
    Returns dict with keys: train, val, test.
    Notes:
      - persistent_workers only valid if num_workers > 0.
      - prefetch_factor only valid if num_workers > 0.
    """

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

    if kmeans:
        kmeans_ds =  SavedEmbDataset(emb_dir=emb_dir, lambdas=lambdas, flag=flag, val_size=val_size, seed=seed, dtype=dtype, split="kmeans")
        return {"kmeans": DataLoader(kmeans_ds, **loader_kwargs)}
    else:
        train_ds = SavedEmbDataset(emb_dir=emb_dir, lambdas=lambdas, flag=flag, val_size=val_size, seed=seed, dtype=dtype, split="train")
        val_ds   = SavedEmbDataset(emb_dir=emb_dir, lambdas=lambdas, flag=flag, val_size=val_size, seed=seed, dtype=dtype, split="val")
        test_ds  = SavedEmbDataset(emb_dir=emb_dir, lambdas=lambdas, flag=flag, val_size=val_size, seed=seed, dtype=dtype, split="test")
        return {
            "train": DataLoader(train_ds, **loader_kwargs),
            "val":   DataLoader(val_ds, **loader_kwargs),
            "test":  DataLoader(test_ds, **loader_kwargs),
        }

def get_reg_dataloaders_from_in_memory_emb_set(
    embeddings: torch.Tensor,
    lambdas: torch.Tensor,
    flag: np.ndarray,
    device: torch.device,
    batch_size: int,
    val_size: float = 0.2,
    seed: int = 42,
    num_workers: int = 0
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

def update_mlm_dataloader(
    dataset: torch.utils.data.Dataset,
    collator: Callable,
    embeddings: torch.Tensor,
    lambdas: torch.Tensor,
    emb_dir: str,
    seed: int = 42,
    n_clusters: int = 512,
    per_device_batch_size_kmeans: int = 1024,
    per_device_batch_size: int = 1024,
    num_workers: int = 2,
    epsilon: int = 2,
    write_to_hard_drive: bool = True,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> DataLoader:
    """Update the order of samples in the dataloader according to informativeness and diversity

    Args:
        embeddings (torch.Tensor). Sequence-level representation.
        lambdas (torch.Tensor). Informativeness of each sequence.
        n_clusters (int): Number of KMeans clusters. Defaults to 4_000.
        seed (int): Random seed. Defaults to 0.
        per_device_batch_size_kmeans (int): Batch size for each GPU when doing clustering.
        
    Returns:
        torch.utils.data.DataLoader
    """
    # shuffle the index list for unconstrained learning, the lambda values do not matter since they are all zeros.
    if epsilon == 1000:
        print("Unconstrained learning - randomize idx order for the next rd")
        updated_idx_order = np.random.permutation(np.arange(len(lambdas)))
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

    kmeans_mdl = MiniBatchKMeans(
        n_clusters=n_clusters, 
        random_state=seed, 
        batch_size=per_device_batch_size_kmeans, 
        n_init='auto',
    )

    if write_to_hard_drive:
        loader = get_reg_dataloaders_from_saved_emb_set(
            emb_dir=emb_dir,
            lambdas=lambdas,
            flag=np.zeros(len(lambdas)), # flag, val_size and seed do not matter here
            val_size=0.0, 
            seed=seed,
            batch_size=per_device_batch_size_kmeans,
            num_workers=num_workers,
            kmeans=True,
            dtype=dtype
        )

        # fit Kmeans
        for emb, _ in loader["kmeans"]:
            kmeans_mdl.partial_fit(emb)

        # prediction
        clusters = []
        for emb, _ in loader["kmeans"]:
            clust_pred = kmeans_mdl.predict(emb)
            clusters.extend(clust_pred)
    else:
        clusters = []
        X = embeddings.detach().to(torch.float32).cpu().numpy()
        clusters = kmeans_mdl.fit_predict(X)

        del X, embeddings
        gc.collect()
    
    # lambdas is global-id axis; align it to the embedding/cluster order (which is idx_order)
    # lambdas_aligned = lambdas.to(torch.float32).numpy()[idx_order]   # now aligned with idx_order

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