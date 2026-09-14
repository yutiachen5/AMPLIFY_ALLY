import os
import numpy as np
import pandas as pd
from typing import List

import torch
from typing import Tuple, List
from torch.utils.data import Dataset

from ..tokenizer import ProteinTokenizer


class InMemoryProteinDataset(Dataset):
    def __init__(
        self,
        paths: dict | None = None,
        path: str | None = None,
        n_partitions: int | None = None,
        max_rows_base_set: int | None = None,
        **kwargs,
    ):
        """
        Protein dataset that loads heldout or base set into memory one at a time, keeping at
        most two sets resident: the current round's pool and the one right
        before it. Later rounds never look further back than that,
        so anything older is evicted as soon as a new set loads.

        Args:
            paths (dict | None): Name -> path, one CSV file per set (heterogeneous tiers).
            path (str | None): Single CSV file to split into `n_partitions` sets by
                round-robin (row i -> partition i % n_partitions) rather than contiguous
                row ranges, so every set is representative of the file's overall length/
                content distribution regardless of on-disk row order (e.g. a file that
                happens to be stored sorted by sequence length). Used instead of `paths`.
            n_partitions (int | None): Number of sets to split `path` into.
            max_rows_base_set (int | None): Cap on how many rows of the first set to load.
        """
        self._n_partitions = n_partitions
        self.samples: dict[int, Tuple[str, str]] = {}

        if path is not None:
            total_rows = sum(1 for _ in open(path, "r")) - 1  # -1 for header
            base_size, remainder = divmod(total_rows, n_partitions)
            self.set_names = [f"partition_{i + 1}" for i in range(n_partitions)]
            self._set_paths: List[str] = [path] * n_partitions
            # Round-robin split: the first `remainder` partitions (rows total_rows -
            # remainder .. total_rows - 1 land there) absorb the one extra row each
            # from the uneven division.
            self.set_lengths: List[int] = [base_size + (1 if i < remainder else 0) for i in range(n_partitions)]
        else:
            self.set_names = list(paths.keys())
            self._set_paths = list(paths.values())
            self.set_lengths = [
                sum(1 for _ in open(p, "r")) - 1  # -1 for header
                for p in self._set_paths
            ]

        self._max_rows_base_set = max_rows_base_set
        if max_rows_base_set is not None:
            self.set_lengths[0] = min(self.set_lengths[0], max_rows_base_set)
        self._cumulative_ends = np.cumsum(self.set_lengths)  # global idx where each set ends

        self._next_set_idx = 0
        self._loaded_from_set_idx = 0  # oldest set index still resident in memory
        self.ensure_loaded_through(1)
        self.idx_order = np.arange(len(self.samples))

    def ensure_loaded_through(self, n_sets: int) -> None:
        """Load sets into memory but retains only the two most recently loaded sets in self.samples."""
        while self._next_set_idx < n_sets:
            set_idx = self._next_set_idx
            start = 0 if set_idx == 0 else int(self._cumulative_ends[set_idx - 1])
            n_rows = self.set_lengths[set_idx]
            file_path = self._set_paths[set_idx]

            with open(file_path, "r") as f:
                next(f)  # skip header
                if self._n_partitions is None:
                    # One CSV per set: read it start to end.
                    for offset, line in zip(range(n_rows), f):
                        row = line.strip().split(",")
                        self.samples[start + offset] = (row[0], row[1])  # (record_id, sequence)
                else:
                    # Shared file, round-robin: row line_idx belongs to set
                    # (line_idx % n_partitions). This set's rows are scattered across
                    # the whole file, so loading it costs a full scan regardless of
                    # which set it is — only matching rows get materialized, though,
                    # so memory stays bounded to this set's share.
                    offset = 0
                    for line_idx, line in enumerate(f):
                        if line_idx % self._n_partitions != set_idx:
                            continue
                        row = line.strip().split(",")
                        self.samples[start + offset] = (row[0], row[1])
                        offset += 1
                        if offset >= n_rows:
                            break
            self._next_set_idx += 1

            # Delete the samples from older set
            keep_from_set = max(0, self._next_set_idx - 2)
            if keep_from_set > self._loaded_from_set_idx:
                evict_before = int(self._cumulative_ends[keep_from_set - 1])
                for k in [k for k in self.samples if k < evict_before]:
                    del self.samples[k]
                self._loaded_from_set_idx = keep_from_set

    def __len__(self):
        return len(self.idx_order)

    def update(self, idx_order):
        self.idx_order = idx_order
        return self

    def __getitem__(self, i: int) -> Tuple[str, str]:
        global_idx = int(self.idx_order[i])
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
                for from_AA, position, to_AA in self.get_mutation_info(mutant):
                    pos_idx = position - 1

                    wt_id = tokenizer.encode(from_AA, add_special_tokens=False)[0]
                    mt_id = tokenizer.encode(to_AA, add_special_tokens=False)[0]
                    pos_to_items.setdefault(pos_idx, []).append((mutant_idx, wt_id, mt_id))

            for pos_idx, values in pos_to_items.items():
                seq_window, new_pos_idx = self.get_sequence_window(target_seq, pos_idx, max_length)
                enc_seq_window = tokenizer.encode(seq_window)
                self.items.append({
                    "dms_idx": dms_id, # DMS_sub_0, ...
                    "pos": pos_idx, # mutant pos
                    "new_pos": new_pos_idx, # mutant pos in seq window
                    "enc_seq": enc_seq_window, # encoded seq window in target seq
                    "mutants": values # [(mutant_idx, wt_id, mt_id), ...]
                })

    def get_sequence_window(self, focus_seq, pos_idx, max_length=512):
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

    def get_mutation_info(self, mutant):
        """Parse 'A42G' or 'A42G:L100V' into [(from_AA, position, to_AA), ...]."""
        mutations = []
        for m in mutant.split(":"):
            mutations.append((m[0], int(m[1:-1]), m[-1]))
        return mutations

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