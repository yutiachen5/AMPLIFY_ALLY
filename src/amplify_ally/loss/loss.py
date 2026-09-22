
import numpy as np
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

from ..tokenizer import ProteinTokenizer


def get_loss(
    device: torch.device,
    reduction: str, 
    vocab_path: str,
    pad_token_id: int,
    mask_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    unk_token_id: int,
    other_special_token_ids: list | None = None,
    label_smoothing: float = 0.0,
    weights: (dict | None) = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.nn.modules.loss._Loss:
    """Public wrapper for constructing the loss function.

    Args:
        device (torch.device): Device.
        vocab_path (str): Path to the vocabulary file to load.
        pad_token_id (int): <PAD> token index.
        mask_token_id (int): <MASK> token index.
        bos_token_id (int): <BOS> token index.
        eos_token_id (int): <EOS> token index.
        unk_token_id (int): <UNK> token index.
        other_special_token_ids (list | None, optional): Indices of the special other tokens. Defaults to None.
        label_smoothing (float, optional): Label smoothing coefficient. Defaults to 0.0.
        weights (dict  |  None, optional): Class weights. Defaults to None.
        strategy_name (str, optional): Strategy for loss manipulation. Defaults to constrained learning.
        dtype (torch.dtype, optional): Dtype of the class_weights. Defaults to torch.float32.

    Returns:
        torch.nn.modules.loss._Loss: A cross-entropy loss function with mean or none reduction.
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

    # Class weights
    class_weights = None
    if weights is not None and any(w != 1 for w in weights.values()):
        class_weights = [weights.get(tokenizer.id_to_token(i), 1) for i in range(len(tokenizer))]
        class_weights = Tensor(class_weights).to(device, dtype, non_blocking=True)

    if reduction == 'none':
        return CrossEntropyLoss(weight=class_weights, reduction="none", ignore_index=-100, label_smoothing=label_smoothing)
    else:
        return CrossEntropyLoss(weight=class_weights, reduction="mean", ignore_index=-100, label_smoothing=label_smoothing)


def get_lagrangian(
    device: torch.device,
    train_loss_seq: torch.Tensor,
    lambdas_current: torch.Tensor,
    epsilon: float = 2.4,
    epsilon_override: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    lambdas_current = lambdas_current.to(device)
    eps = epsilon_override.to(device) if epsilon_override is not None else epsilon

    lagrangian = (train_loss_seq * (1 + lambdas_current) - lambdas_current * eps).nanmean()
    constraint_violations = (train_loss_seq - eps).nanmean().item()

    return lagrangian, constraint_violations


def update_dual_variables(
    train_loss_seq: torch.Tensor,
    lambdas_current: torch.Tensor,
    lr_dual: float = 0.1,
    dtype: torch.dtype = torch.float32,
    epsilon: float = 2.4,
    epsilon_override: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:

    train_loss_seq = train_loss_seq.detach().cpu().to(dtype)
    eps = epsilon_override.to(dtype) if epsilon_override is not None else epsilon
    nan_idxs = torch.nonzero(torch.isnan(train_loss_seq), as_tuple=True)
    train_loss_seq[nan_idxs] = eps[nan_idxs] if torch.is_tensor(eps) else eps

    lambdas_current += lr_dual * (train_loss_seq - eps)
    lambdas_current.data.clamp_(min=0)

    return lambdas_current


class LengthBinnedEpsilon:
    """Nonparametric, length-conditional loss threshold for dual ascent.

    A single scalar epsilon makes the dual variable track absolute
    difficulty, which is confounded with sequence length (see
    project_length_confound_finding). This replaces it with a per-length-bin
    threshold refit after each round from that round's own real (first-visit)
    losses, so a sample only accrues lambda when its loss is worse than
    typical for other samples of similar length trained around the same
    point in training — not just whenever it's long or short in absolute
    terms.

    Bin edges are fixed the first time `update` is called (quantiles of that
    round's lengths) and reused for every later round, so lookups stay
    well-defined even for a round whose own length distribution differs.
    Bin thresholds are refit every round from only the most recently
    completed round's data (not accumulated across all history), since the
    model's overall calibration keeps shifting round to round and a stale
    threshold would silently drift out of step with it.

    Without a margin, a bin's threshold would sit right at that bin's own
    just-observed mean loss — so by construction, roughly half of next
    round's samples land above it and half below, and the average
    constraint violation self-cancels to ~0 every round regardless of how
    training is actually going. That starves the dual variable of the
    sustained, one-directional pressure it needs to grow to a meaningful
    magnitude. `margin` is calibrated once, from round 1's own average
    violation under the cold-start scalar epsilon (the same gap the
    original fixed-epsilon design relied on), and held fixed thereafter —
    so later per-bin thresholds sit that same amount below their own mean,
    preserving comparable sustained pressure instead of collapsing to zero
    the moment thresholds become each bin's own recent average.
    """

    def __init__(self, global_epsilon: float, n_bins: int = 50, min_count: int = 50):
        self.global_epsilon = global_epsilon
        self.n_bins = n_bins
        self.min_count = min_count
        self.bin_edges: np.ndarray | None = None  # quantile cut points, set on first update()
        self.margin: float | None = None  # calibrated once, from the first update() call
        self.bin_epsilon = np.full(n_bins, global_epsilon, dtype=np.float64)

    def _bin_index(self, lengths: np.ndarray) -> np.ndarray:
        if self.bin_edges is None:
            return np.zeros(len(lengths), dtype=np.int64)
        return np.searchsorted(self.bin_edges, lengths, side="right").clip(0, self.n_bins - 1)

    def lookup(self, lengths: np.ndarray) -> torch.Tensor:
        idx = self._bin_index(np.asarray(lengths))
        return torch.as_tensor(self.bin_epsilon[idx], dtype=torch.float32)

    def update(self, lengths: np.ndarray, losses: np.ndarray) -> None:
        lengths = np.asarray(lengths)
        losses = np.asarray(losses)
        finite = np.isfinite(losses)
        lengths, losses = lengths[finite], losses[finite]
        if len(losses) == 0:
            return

        if self.bin_edges is None:
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.bin_edges = np.quantile(lengths, quantiles)

        if self.margin is None:
            # Calibrate once, from this (first) round's own average violation under the
            # cold-start scalar epsilon. Clamped to >= 0: a negative value would mean
            # round 1 didn't violate on average (epsilon too loose), and subtracting a
            # negative margin would raise the bar above the mean instead of below it.
            self.margin = max(0.0, float(losses.mean() - self.global_epsilon))

        idx = self._bin_index(lengths)
        global_mean = losses.mean()
        for b in range(self.n_bins):
            mask = idx == b
            n = int(mask.sum())
            if n == 0:
                continue
            cell_mean = losses[mask].mean()
            # Empirical-Bayes shrinkage toward this round's global mean for sparse bins,
            # so a handful of samples in a rarely-hit length bin can't swing epsilon wildly.
            shrunk_mean = (n * cell_mean + self.min_count * global_mean) / (n + self.min_count)
            self.bin_epsilon[b] = shrunk_mean - self.margin

    def state_dict(self) -> dict:
        return {"bin_edges": self.bin_edges, "bin_epsilon": self.bin_epsilon, "margin": self.margin}

    def load_state_dict(self, state: dict) -> None:
        self.bin_edges = state["bin_edges"]
        self.bin_epsilon = state["bin_epsilon"]
        self.margin = state["margin"]