import gc
from collections import defaultdict
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from ..dataset import get_reg_dataloaders_from_saved_emb_set


class KMeansTrainer:
    """
    Train MiniBatchKMeans and update the MLM dataloader order according to
    diversity (cluster) and informativeness (lambda).

    Notes
    -----
    - If write_to_hard_drive=True, embeddings are streamed from disk using
      get_reg_dataloaders_from_saved_emb_set. Only use a subset of emb to train,
      and expand prediction to the all samples due to time constraint.
    - If fit_subset_size is not None, only a subset is used for k-means fitting,
      but prediction is still done on the full dataset.
    - idx_order is assumed to define the embedding order for the current round.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        collator: Callable,
        embeddings: torch.Tensor,
        idx_order: np.ndarray,
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
        fit_subset_size: Optional[int] = None,
        kmeans_n_init="auto",
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        **kwargs,
    ):
        self.dataset = dataset
        self.collator = collator
        self.emb_dir = emb_dir

        self.seed = seed
        self.n_clusters = n_clusters
        self.per_device_batch_size_kmeans = per_device_batch_size_kmeans
        self.per_device_batch_size = per_device_batch_size
        self.num_workers = num_workers
        self.epsilon = epsilon
        self.write_to_hard_drive = write_to_hard_drive
        self.dtype = dtype
        self.fit_subset_size = fit_subset_size
        self.kmeans_n_init = kmeans_n_init

        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.kwargs = kwargs

        self.embeddings = embeddings
        self.idx_order = idx_order
        self.lambdas = lambdas

        self.kmeans_mdl = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=per_device_batch_size_kmeans,
            n_init=kmeans_n_init,
        )

    def _sample_batch_for_fit(
        self,
        X: np.ndarray,
        seen_total: int, # num of embeddings the loader has already read past
        total_n: int, # total num of points
        taken_total: int, # num of points which have actually been kept and used for partial_fit
        fit_target: int,
        rng: np.random.Generator,
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Sample approximately the right number from this batch so the total
        fit subset lands near fit_target over the full stream.
        """
        B = len(X) # batch size
        remaining_points = total_n - seen_total
        remaining_to_take = fit_target - taken_total

        if B == 0 or remaining_to_take <= 0 or remaining_points <= 0:
            return None, 0

        expected_take = int(round(B * (remaining_to_take / remaining_points)))
        n_take = max(0, min(B, expected_take))

        # last-batch safeguard
        if remaining_points == B and B >= remaining_to_take:
            n_take = remaining_to_take

        if n_take == 0:
            return None, 0

        if n_take == B:
            return X, B

        idx = rng.choice(B, size=n_take, replace=False)
        return X[idx], n_take

    def _fit_predict_from_saved_emb(
        self,
        lambdas: torch.Tensor,
    ) -> np.ndarray:
        """
        Stream embeddings from disk:
        - fit MiniBatchKMeans on full stream or subset
        - predict clusters on full stream
        """
        N = len(lambdas)
        loader = get_reg_dataloaders_from_saved_emb_set(
            emb_dir=self.emb_dir,
            lambdas=lambdas,
            flag=np.zeros(len(lambdas), dtype=np.int8),  # unused in kmeans path
            val_size=0.0,
            seed=self.seed,
            batch_size=self.per_device_batch_size_kmeans,
            num_workers=self.num_workers,
            kmeans=True,
            dtype=self.dtype,
        )

        self.kmeans_mdl = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            batch_size=self.per_device_batch_size_kmeans,
            n_init=self.kmeans_n_init,
        )

        fit_target = None
        if self.fit_subset_size is not None:
            fit_target = int(min(max(1, self.fit_subset_size), N))

        rng = np.random.default_rng(self.seed)

        # -------------------------
        # fit on subset
        # -------------------------
        pbar = tqdm(
            desc="KMeans fit",
            unit="batch",
            initial=0,
            total=len(loader["kmeans"]),
        )

        seen_total = 0
        taken_total = 0

        for emb, _ in loader["kmeans"]:
            if fit_target is None:
                self.kmeans_mdl.partial_fit(emb)
            else:
                X_sub, n_take = self._sample_batch_for_fit(
                    X=emb,
                    seen_total=seen_total,
                    total_n=N,
                    taken_total=taken_total,
                    fit_target=fit_target,
                    rng=rng,
                )
                if X_sub is not None and len(X_sub) > 0:
                    self.kmeans_mdl.partial_fit(X_sub)
                    taken_total += n_take

            seen_total += len(emb)
            pbar.update(1)

            if fit_target is not None and taken_total >= fit_target:
                break

        pbar.close()

        # -------------------------
        # predict on all
        # -------------------------
        clusters = np.empty(N, dtype=np.int32)
        offset = 0

        pbar = tqdm(
            desc="KMeans predict",
            unit="batch",
            initial=0,
            total=len(loader["kmeans"]),
        )

        for emb, _ in loader["kmeans"]:
            pred = self.kmeans_mdl.predict(emb)
            clusters[offset:offset + len(pred)] = pred
            offset += len(pred)
            pbar.update(1)

        pbar.close()

        return clusters

    def _fit_predict_from_memory(
        self,
        embeddings: torch.Tensor,
    ) -> np.ndarray:
        """
        Use in-memory embeddings directly.
        """
        clusters = self.kmeans_mdl.fit_predict(embeddings)
        del embeddings
        gc.collect()
        return clusters

    def _reorder_indices(
        self,
        clusters: np.ndarray,
        lambdas: torch.Tensor,
        idx_order: np.ndarray,
    ) -> np.ndarray:
        """
        Sort by:
        1. cluster id ascending
        2. lambda descending within cluster

        Then interleave samples across clusters.
        """
        lambdas_aligned = lambdas.detach().to(torch.float32).cpu().numpy()[idx_order] # float32 just for sorting

        sorted_triplets = sorted(
            zip(clusters, lambdas_aligned, idx_order),
            key=lambda t: (t[0], -t[1]),
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
            for c in range(self.n_clusters):
                if i < len(cluster_to_samples[c]):
                    _, idx = cluster_to_samples[c][i]
                    updated_idx_order.append(idx)

        updated_idx_order = np.array(updated_idx_order, dtype=idx_order.dtype)

        del cluster_to_samples
        gc.collect()

        return updated_idx_order

    def _build_train_dataloader(self, updated_idx_order: np.ndarray) -> DataLoader:
        loader_kwargs = dict(
            dataset=self.dataset.update(updated_idx_order),
            batch_size=self.per_device_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**loader_kwargs)

    def train_and_update_dataloader(self) -> Tuple[np.ndarray, DataLoader]:
        """
        Main entry point:
        - unconstrained random reorder if epsilon == 1000
        - else kmeans fit/predict
        - then reorder according to diversity + informativeness
        """
        if self.epsilon == 1000:
            print("Unconstrained learning - randomize idx order for the next rd")
            updated_idx_order = np.random.permutation(self.idx_order)
            return updated_idx_order, self._build_train_dataloader(updated_idx_order)

        if self.write_to_hard_drive:
            clusters = self._fit_predict_from_saved_emb(lambdas=self.lambdas)
        else:
            clusters = self._fit_predict_from_memory(embeddings=self.embeddings)

        updated_idx_order = self._reorder_indices(
            clusters=clusters,
            lambdas=self.lambdas,
            idx_order=self.idx_order,
        )

        return updated_idx_order, self._build_train_dataloader(updated_idx_order)