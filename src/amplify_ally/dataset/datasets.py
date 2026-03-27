import torch
from typing import Tuple, Iterator
from itertools import islice, zip_longest, repeat, chain
from torch.utils.data import IterableDataset, get_worker_info, Dataset

from typing import List

import os
import random
import hashlib
import numpy as np


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
    def __init__(self, X_train, X_val, y_train, y_val, X_test, split="train"):
        self.split = split

        if split == "train":
            self.x_data, self.y_data = X_train, y_train
        elif split == "val":
            self.x_data, self.y_data = X_val, y_val
        elif split == "test":
            self.x_data = X_test
    
    def __getitem__(self, i):
        if self.split == "train" or self.split == "val":
            return self.x_data[i], self.y_data[i]
        elif self.split == "test":
            return self.x_data[i]

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
        split: str = "train",
        shard_size: int = 1_000_000,
        **kwargs,
    ):
        self.emb_dir = emb_dir
        self.lambdas = lambdas
        self.split = split
        self.dtype = dtype
        self.shard_size = shard_size

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
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return int(len(self.active_idx))

    def _load_emb(self, global_id: int):
        shard_id  = global_id // self.shard_size
        local_idx = global_id % self.shard_size

        emb_path = os.path.join(self.emb_dir, f"shard_{shard_id:04d}.npy")
        ids_path = os.path.join(self.emb_dir, f"shard_{shard_id:04d}_ids.npy")

        ids = np.load(ids_path)
        assert ids[local_idx] == global_id, (
            f"ID mismatch at shard={shard_id} local_idx={local_idx}: "
            f"expected {global_id}, got {ids[local_idx]}."
        )

        emb = np.load(emb_path, mmap_mode="r")[local_idx].copy()

        if self.split != "kmeans":
            return torch.from_numpy(emb).to(dtype=self.dtype)

        return emb

    def __getitem__(self, i: int):
        global_id = int(self.active_idx[i])
        emb = self._load_emb(global_id)
        l = self.lambdas[global_id]
        
        return emb, l