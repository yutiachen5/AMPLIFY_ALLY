import torch
from torch.utils.data import DataLoader

from ..tokenizer import ProteinTokenizer

from .iterable_protein_dataset import IterableProteinDataset, InMemoryProteinDataset
from .data_collator import DataCollatorMLM

import math
from sklearn.cluster import MiniBatchKMeans

from typing import List


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
            per_device_batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
        )
    else:
        return {
            k: DataLoader(
                InMemoryProteinDataset([v]),
                per_device_batch_size,
                collate_fn=collator,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
                persistent_workers=True,
            )
            for k, v in paths.items()
        }

    
def update_dateloader(
    protein_dataset: torch.utils.data.Dataset,
    embeddings: torch.Tensor,
    lambdas: torch.Tensor,
    global_map: List[tuple[int,int]],
    n_clusters: int,
    space_in_cluster: int,
    seed: int,
    per_device_batch_size: int,
    paths: dict
) -> DataLoader:
    """Get the embeddinds after each round

    Args:
        embeddings (torch.Tensor). Sequence-level representation.
        lambdas (torch.Tensor). Informativeness of each sequence.
        global_map (List[tuple[int,int]]). Gobal mapping of training samples (file_id, line_number).
        n_clusters (int): Number of KMeans clusters. Defaults to 4_000.
        space_in_cluster: Quota for each cluster. Defaults to 10_000.
        seed (int): Random seed. Defaults to 0.
        per_device_batch_size (int): Batch size for extracting embeddings. Defaults to 64.
        
    Returns:
        List[]: cluster id of each sample in the training dataloader
    """
    clusters = []
    idxs_within_quota, lambdas_within_quota = [], []
    idxs_over_quota, lambdas_over_quota = [], []

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, 
        random_state=seed, 
        batch_size=per_device_batch_size, 
        n_init='auto',
    )

    clusters = kmeans.fit_predict(embeddings)
    sorted_triplets = sorted(zip(clusters, lambdas, global_map), key=lambda x: (x[0], -x[1])) # sort by cluster and then lambda
    sorted_clusters, sorted_lambdas, sorted_idxs = zip(*sorted_triplets)

    # for i, idx in enumerate(sorted_idxs):
    #     cluster_id = sorted_clusters[i]
    #     lambda_value = sorted_lambdas[i]

    #     if space_in_cluster[cluster_id] > 0:
    #         idxs_within_quota.append(idx)
    #         lambdas_within_quota.append(lambda_value)
    #         space_in_cluster[cluster_id] -= 1
    #     else: 
    #         idxs_over_quota.append(idx)
    #         lambdas_over_quota.append(lambda_value)

    # idx_order = idxs_within_quota + idxs_over_quota
    # lambda_order = lambdas_within_quota + lambdas_over_quota

    for i in range(math.ceil(len(lambdas)/n_clusters)): # 128 samples in a batch --> 128 cluster
        for j in range(n_clusters):
            idx_order.append(sorted_idxs[i])
            lambdas_order.append(sorted_lambdas[i])
    
    protein_dataset.update(idx_order)

    return DataLoader(
        protein_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

# go through by clusters
# batch size = n_clusters = 128


