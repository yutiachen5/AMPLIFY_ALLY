from typing import Tuple, Iterator
from itertools import islice, zip_longest, repeat, chain
from torch.utils.data import IterableDataset, get_worker_info, Dataset

from typing import List

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

    def parse_file(self) -> Iterator[Tuple[int, int, str, str]]:
        worker_info = get_worker_info()
        step = 1 if worker_info is None else worker_info.num_workers
        offset = 0 if worker_info is None else worker_info.id

        files, iterators = [], []
        for file_id, (path, n) in enumerate(zip(self.paths, self.samples_per_set)):
            f = open(path, "r")
            next(f)  # skip header
            files.append(f)

            # Wrap each line with file_id and line_number
            def line_enumerator(file, file_id):
                for line_number, row in enumerate(file):
                    yield file_id, line_number, row

            iterators.extend(repeat(line_enumerator(f, file_id), n))

        final_iterator = chain.from_iterable(zip_longest(*iterators, fillvalue=None))

        for row in islice(final_iterator, offset, None, step):
            if row is not None:
                file_id, line_number, raw = row
                record_id, sequence = raw.strip().split(",")
                yield file_id, line_number, record_id, sequence

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

    def __getitem__(self, i: int) -> Tuple[str, str]:
        """
        Fetch a sample by index.

        Returns:
            (record_id, sequence)
        """
        global_idx = self.idx_order[i]
        sample = self.samples[global_idx]
        return global_idx, sample[0], sample[1] # (record_id, sequence)