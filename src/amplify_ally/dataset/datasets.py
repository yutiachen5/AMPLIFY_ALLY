import torch
from typing import Tuple, Iterator, List
from itertools import islice, zip_longest, repeat, chain
from torch.utils.data import IterableDataset, get_worker_info, Dataset

from typing import List

import os
import random
import hashlib
import numpy as np
import pandas as pd

from ..tokenizer import ProteinTokenizer


class IterableProteinDataset(IterableDataset):
    def __init__(self, paths: list, samples_before_next_set: list | None):
        """An iterable dataset that reads protein sequences from a file.

        Args:
            paths (list): Paths to the CSV files to read.
            samples_before_next_set (list | None): Number of samples of each dataset to return before moving to the
            next dataset (interleaving).
        """
        self.paths = paths
        self.samples_per_set = samples_before_next_set if samples_before_next_set is not None else [1] * len(paths)

    def parse_file(self) -> str:
        worker_info = get_worker_info()
        step = 1 if worker_info is None else worker_info.num_workers
        offset = 0 if worker_info is None else worker_info.id

        files, iterator = [], []
        for path, n in zip(self.paths, self.samples_per_set):
            # Open the file
            file = open(path, "r")
            # Skip header
            next(file)
            # Add the file to the list of files to close them at the end
            files.append(file)
            # Add the file iterator to the list of iterators n times
            iterator.extend(repeat(file, n))

        # Interleave the iterators and pad with None
        iterator = chain.from_iterable(zip_longest(*iterator, fillvalue=None))

        # Iterate through the datasets
        for row in islice(iterator, offset, None, step):
            if row is not None:
                # Assumes (record_id,sequence)
                yield row.strip().split(",")

        # Closing the files
        for file in files:
            file.close()

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return self.parse_file()


class InMemoryProteinDataset(Dataset):
    def __init__(self, paths: dict, **kwargs):
        """
        Protein dataset that loads all data into memory.

        Args:
            paths (list): Paths to the CSV files to read.
        """
        self.paths = paths
        self.samples: List[Tuple[str, str]] = []

        # Load all sequences into memory
        for path in self.paths:
            with open(path, "r") as f:
                next(f)  # skip header
                for line in f:
                    row = line.strip().split(",")
                    self.samples.append((row[0], row[1]))  # (record_id, sequence)
        self.idx_order = np.arange(len(self.samples))
                    
    def __len__(self):
        return len(self.samples)

    def update(self, idx_order):
        self.idx_order = idx_order
        return self

    def __getitem__(self, i: int) -> Tuple[str, str]:
        global_idx = self.idx_order[i]
        sample = self.samples[global_idx]
        return global_idx, sample[0], sample[1] # (record_id, sequence)


class InMemoryEmbDataset(Dataset): 
    def __init__(self, x_data, y_data, split="train"):
        self.split = split
        self.x_data, self.y_data = x_data, y_data

    def __getitem__(self, i):
        if self.split == "train" or self.split == "val":
            return self.x_data[i], self.y_data[i]
        elif self.split == "test":
            return self.x_data[i], torch.tensor(float('nan'))

    def __len__(self):
        return self.x_data.shape[0]


class SavedEmbDataset(Dataset):
    def __init__(
        self,
        emb_dir: str,
        lambdas: torch.Tensor,
        flag: np.ndarray,
        dtype: torch.dtype = torch.float32,
        val_size: float = 0.2,
        seed: int = 42,
        shard_size: int = 1_000_000,
        id_to_loc: dict | None = None,
        split: str = "train",
        **kwargs,
    ):
        self.emb_dir = emb_dir
        self.lambdas = lambdas
        self.split = split
        self.dtype = dtype
        self.shard_size = shard_size
        self.id_to_loc = id_to_loc

        n = len(lambdas)

        if split == "kmeans":
            self.active_idx = np.arange(n, dtype=np.int64)
            return

        test_idx = np.flatnonzero(flag < 1).astype(np.int64)
        train_val_idx = np.flatnonzero(flag >= 1).astype(np.int64)

        rng = np.random.default_rng(seed)
        n_val = int(round(len(train_val_idx) * val_size))

        perm = rng.permutation(len(train_val_idx))
        val_pos = perm[:n_val]
        train_pos = perm[n_val:]

        val_idx = train_val_idx[val_pos]
        train_idx = train_val_idx[train_pos]

        if split == "train":
            self.active_idx = train_idx
        elif split == "val":
            self.active_idx = val_idx
        elif split == "test":
            self.active_idx = test_idx

    def __len__(self):
        return int(len(self.active_idx))

    def _load_emb(self, global_id: int):
        shard_id, local_id = self.id_to_loc[global_id]
        emb_path = os.path.join(self.emb_dir, f"shard_{shard_id:04d}.npy")
        ids = np.load(ids_path)

        assert ids[local_id] == global_id, (
            f"ID mismatch at shard={shard_id} local_id={local_id}: "
            f"expected {global_id}, got {ids[local_id]}."
        )        

        emb = np.load(emb_path, mmap_mode="r")[local_id].copy()
        if self.split != "kmeans":
            return torch.from_numpy(emb).to(dtype=self.dtype)
        return emb

    def __getitem__(self, i: int):
        global_id = int(self.active_idx[i])
        emb = self._load_emb(global_id)
        l = self.lambdas[global_id]
        
        return emb, l


def get_sequence_window(focus_seq, pos_idx, max_length=512):
    """Extract a window of length max_length centered around pos_idx.
    If the sequence is shorter than max_length, return the full sequence.
    """
    seq_len = len(focus_seq)
    if seq_len <= max_length:
        return focus_seq, pos_idx
 
    half  = max_length // 2
    start = max(0, pos_idx - half)
    end   = start + max_length
 
    if end > seq_len: 
        end   = seq_len
        start = end - max_length
 
    return focus_seq[start:end], pos_idx - start
 
 
def get_mutation_info(mutant):
    """Parse 'A42G' or 'A42G:L100V' into [(from_AA, position, to_AA), ...]."""
    mutations = []
    for m in mutant.split(":"):
        mutations.append((m[0], int(m[1:-1]), m[-1]))
    return mutations

class ProteinGymDataset(Dataset):
    """Each item is one unique masked position across all assays.
 
    The dataloader batches these items together. After the forward pass,
    log-probs are fanned out to every mutant that references that position.
 
    Args:
        DMS_reference_file_path: Path to DMS_substitutions.csv.
        DMS_data_dir: Path to folder with per-assay CSV files.
        tokenizer: Tokenizer with encode() and mask_token_id.
        max_length: Maximum sequence length; longer sequences are windowed.
        excluded_indices: Set of DMS integer indices to skip.
    """
 
    def __init__(
        self, 
        DMS_reference_file_path: str,
        DMS_data_dir: str, 
        tokenizer: ProteinTokenizer, 
        max_length: int =512,
        excluded_indices: list | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_tok_id = tokenizer.mask_token_id
 
        mapping = pd.read_csv(DMS_reference_file_path)
        assay_ids = [i for i in range(len(mapping)) if i not in excluded_indices]

        self.items = []
        self.assay_scores = {}

        for i in assay_ids:
            row = mapping.iloc[i]
            dms_id = row["DMS_id"]
            dms_file = row["DMS_filename"]
            target_seq = row["target_seq"].upper()
            dms_path = os.path.join(DMS_data_dir, dms_file)

            dms_data = pd.read_csv(dms_path, low_memory=False)
            self.assay_scores[dms_id] = {
                "dms_scores": dms_data["DMS_score"].values.copy(),
                "model_scores": np.zeros(len(dms_data)),
            }

            pos_to_items = {} # pos_idx -> [(mutant_idx1, wt_id1, my_id1), (mutant_idx2, wt_id2, my_id2), ...]
            for mutant_idx, mutant in enumerate(dms_data["mutant"].values):
                for from_AA, position, to_AA in get_mutation_info(mutant):
                    pos_idx = position - 1

                    wt_id = tokenizer.encode(from_AA, add_special_tokens=False)[0]
                    mt_id = tokenizer.encode(to_AA, add_special_tokens=False)[0]
                    pos_to_items.setdefault(pos_idx, []).append((mutant_idx, wt_id, mt_id))

            for pos_idx, values in pos_to_items.items():
                seq_window, new_pos_idx = get_sequence_window(target_seq, pos_idx, max_length)
                enc_seq_window = tokenizer.encode(seq_window)
                self.items.append({
                    "dms_idx": dms_id, # DMS_sub_0, ...
                    "pos": pos_idx, # mutant pos
                    "new_pos": new_pos_idx, # mutant pos in seq window
                    "enc_seq": enc_seq_window, # encoded seq window in target seq
                    "mutants": values # [(mutant_idx, wt_id, mt_id), ...]
                })

    def __len__(self):
        return len(self.items)
 
    def __getitem__(self, idx):
        item = self.items[idx]

        # mask encoded seq for model input
        enc_seq = torch.as_tensor(item["enc_seq"], dtype=torch.long).clone().detach()
        mask_pos = item["new_pos"]
        enc_seq[mask_pos + 1] = self.mask_tok_id # +1 for BOS
        return {
            "masked_ids": enc_seq, # (max_len,)
            "new_pos": item["new_pos"], 
            "dms_idx": item["dms_idx"], 
            "mutants": item["mutants"], # list 
        }