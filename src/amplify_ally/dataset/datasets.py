import os
import numpy as np
import pandas as pd
from typing import List

import torch
from typing import Tuple, List
from torch.utils.data import Dataset

from ..tokenizer import ProteinTokenizer


class InMemoryProteinDataset(Dataset):
    def __init__(self, paths: dict, max_rows_base_set: int | None = None, **kwargs):
        """
        Protein dataset that loads heldout or base set into memory one at a time, keeping at
        most two sets resident: the current round's pool and the one right
        before it. Later rounds never look further back than that, 
        so anything older is evicted as soon as a new set loads.

        Args:
            paths (dict): Name -> path to the CSV files to read.
            max_rows_base_set (int | None): Cap on how many rows of the base set to load. 
        """
        self._set_paths: List[Tuple[str, str]] = list(paths.items())  # [(name, path), ...]
        self.set_names: List[str] = [name for name, _ in self._set_paths]
        self.samples: dict[int, Tuple[str, str]] = {}

        self.set_lengths: List[int] = [
            sum(1 for _ in open(path, "r")) - 1  # -1 for header
            for _, path in self._set_paths
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
            start = 0 if self._next_set_idx == 0 else int(self._cumulative_ends[self._next_set_idx - 1])
            row_cap = self._max_rows_base_set if self._next_set_idx == 0 else None
            _, path = self._set_paths[self._next_set_idx]
            with open(path, "r") as f:
                next(f)  # skip header
                for offset, line in enumerate(f):  # offset: row counter within each set 
                    if row_cap is not None and offset >= row_cap:
                        break
                    row = line.strip().split(",")
                    self.samples[start + offset] = (row[0], row[1])  # (record_id, sequence)
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