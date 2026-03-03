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


# class EmbDataset(Dataset):
#     def __init__(self, emb):
#         self.emb = emb

#     def __len__(self):
#         return len(self.emb)

#     def __getitem__(self, i):
#         return self.emb[i]


class SavedEmbDataset(Dataset):
    def __init__(
        self,
        emb_dir: str,
        lambdas: torch.Tensor,
        flag: np.ndarray,
        val_size: float = 0.2,
        seed: int = 42,
        split: str = "train",
        **kwargs,
    ):
        self.emb_dir = emb_dir
        self.lambdas = lambdas
        idx = np.arange(len(lambdas))

        if split == "kmeans":
            self.active_idx = idx
        else:
            test_idx = idx[flag < 1]
            train_val_idx = idx[flag >= 1]

            rng = np.random.default_rng(seed)
            n_val = int(round(len(train_val_idx) * val_size))
            n_val = min(n_val, len(train_val_idx))

            val_idx = rng.choice(train_val_idx, size=n_val, replace=False)
            train_idx = np.array([j for j in train_val_idx if j not in set(val_idx.tolist())], dtype=np.int64)

            if split == "train":
                self.active_idx = train_idx
            elif split == "val":
                self.active_idx = val_idx
            elif split == "test":
                self.active_idx = test_idx

    def __len__(self):
        return int(len(self.active_idx))

    def _load_emb(self, global_id: int) -> torch.Tensor:
        emb_path = os.path.join(self.emb_dir, f"seq_{int(global_id)}.pt")
        emb = torch.load(emb_path, map_location="cpu")  
        return emb

    def __getitem__(self, i: int):
        global_id = int(self.active_idx[i])

        emb = self._load_emb(global_id)
        l = self.lambdas[global_id] 

        return emb, l