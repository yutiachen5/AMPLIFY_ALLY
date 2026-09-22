import torch
from torch import Tensor
from typing import List, Tuple

from ..tokenizer import ProteinTokenizer


class DataCollatorMLM(object):
    def __init__(
        self,
        tokenizer: ProteinTokenizer,
        max_length: int,
        random_truncate: bool,
        return_labels: bool,
        mask_probability: float,
        span_probability: float,
        span_max: int,
        exclude_special_tokens_replacement: bool,
        padding: str,
        pad_to_multiple_of: int,
        dtype: torch.dtype,
        deterministic_seed: int | None = None,
        **kwargs,
    ) -> None:
        """Data collator used for masked language modeling and span masking.

        Args:
            deterministic_seed (int | None, optional): If set, every random draw for a
                sample (truncation offset, mask positions, MASK/random-word replacement)
                is seeded from `deterministic_seed + global_id` instead of the global RNG
                state — so the same physical tokens get masked for a given sample every
                time it's evaluated, regardless of run, worker count, or how many times
                it's evaluated. Intended for validation only; leave None for training so
                masking stays a stochastic augmentation across epochs. Only covers the
                plain-MLM masking path, not span masking (span_probability > 0).
        """
        # Tokenizer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.random_truncate = random_truncate
        self.deterministic_seed = deterministic_seed

        # Return labels (not compatible with accelerate)
        self.return_labels = return_labels

        # MLM
        self.mask_probability = mask_probability

        # Span mask
        self.span_probability = span_probability
        self.span_max = span_max

        # Random words for the replacements
        self.replacement_ids = torch.ones((len(self.tokenizer)))
        if exclude_special_tokens_replacement:
            for i in self.tokenizer.special_token_ids:
                self.replacement_ids[i] = 0

        # Pad the sequence to max_length or the longest sequence in the batch
        self.padding = padding

        # Pad the sequence to a multiple of the provided value
        self.pad_to_multiple_of = pad_to_multiple_of

        # Dtype
        self.dtype = dtype

    def __call__(self, inputs: List[Tuple]) -> Tuple[Tensor, Tensor, Tensor]:
        """Prepare masked tokens for masked language modeling (80% MASK, 10% random, 10% original).
        Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.

        Inspired by https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/data/data_collator.py#L607
        """
        # Unpack
        global_id, labels, proteins = zip(*inputs)

        # One generator per sample, seeded from its global_id, reused for every random
        # draw below (truncation, mask positions, replacement decisions, word choice) in
        # a fixed call order — so a given sample's mask is identical run to run, worker
        # to worker, regardless of what else is going on in the process's RNG state.
        generators = (
            [torch.Generator().manual_seed(self.deterministic_seed + int(gid)) for gid in global_id]
            if self.deterministic_seed is not None else None
        )

        global_id = torch.tensor(global_id, dtype=torch.long)

        # Tokenize the inputs
        if generators is not None:
            proteins = [
                self.tokenizer.encode(p, self.max_length, random_truncate=self.random_truncate, generator=g)
                for p, g in zip(proteins, generators)
            ]
        else:
            proteins = [self.tokenizer.encode(p, self.max_length, random_truncate=self.random_truncate) for p in proteins]

        # Compute the length of the batch min(longest sequence, max_length)
        if self.padding == "longest":
            max_length = max(p.size(0) for p in proteins)
        elif self.padding == "max_length":
            max_length = self.max_length

        # Pad the sequence to a multiple of the provided value. Necessary for memory efficient attention
        if self.pad_to_multiple_of is not None and (max_length % self.pad_to_multiple_of != 0):
            max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        # Initialize the input tensor with padding and the padding mask with False
        x = torch.full(size=(len(proteins), max_length), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        masked_ids = torch.full(size=(len(proteins), max_length), fill_value=False, dtype=torch.bool)

        # Stack the input tensors
        for i, p in enumerate(proteins):
            x[i, : p.shape[0]] = p

        # Compute the padding, <bos> and <eos> masks
        pad_mask = x == self.tokenizer.pad_token_id
        bos_mask = x == self.tokenizer.bos_token_id
        eos_mask = x == self.tokenizer.eos_token_id

        # No masking
        if self.mask_probability is None or self.mask_probability == 0:
            # Replace masked position with float(-inf)
            pad_mask = torch.where(pad_mask, float("-inf"), float(0.0)).type(self.dtype)

            # Output is full -100 (ignore_index)
            y = torch.full(x.shape, -100)

            return (global_id, labels, x, y, pad_mask) if self.return_labels else (global_id, x, y, pad_mask)

        # MLM
        if self.span_probability is None or self.span_max is None or self.span_probability == 0 or self.span_max == 1:
            probability_matrix = torch.full(x.shape, self.mask_probability)
            probability_matrix.masked_fill_(pad_mask | bos_mask | eos_mask, value=0.0)
            if generators is not None:
                masked_ids = torch.stack([
                    torch.bernoulli(probability_matrix[i], generator=g).bool() for i, g in enumerate(generators)
                ])
            else:
                masked_ids = torch.bernoulli(probability_matrix).bool()

        # Span masking
        else:
            for i, p in enumerate(proteins):
                uniform_dist = torch.distributions.uniform.Uniform(0, p.size(0))
                geometric_dist = torch.distributions.geometric.Geometric(self.span_probability)
                while torch.sum(masked_ids[i]) / p.size(0) < self.mask_probability:
                    span_start = int(uniform_dist.sample().item())
                    span_length = int(min(geometric_dist.sample().item(), self.span_max - 1, p.size(0) - span_start))
                    masked_ids[i, span_start : span_start + span_length + 1] = True
                    # Unmask the padding, <bos> and <eos> tokens (note that padding should not be necessary)
                    masked_ids[i] = masked_ids[i] & ~pad_mask[i] & ~bos_mask[i] & ~eos_mask[i]

        # Create the label tensor
        y = x.clone()

        # Only compute the loss on the masked tokens (-100 is the default ignore_index of PyTorch)
        y[~masked_ids] = -100

        if generators is not None:
            # 80% of the time, the masked input tokens are replaced with <MASK>
            replaced_ids = torch.stack([
                torch.bernoulli(torch.full(y.shape[1:], 0.8), generator=g).bool() for g in generators
            ]) & masked_ids
            x[replaced_ids] = self.tokenizer.mask_token_id

            # 10% of the time, the masked input tokens are replaced with a random word
            random_ids = torch.stack([
                torch.bernoulli(torch.full(y.shape[1:], 0.5), generator=g).bool() for g in generators
            ]) & masked_ids & ~replaced_ids
            random_words = torch.stack([
                torch.multinomial(self.replacement_ids, x.shape[1], replacement=True, generator=g) for g in generators
            ])
            x[random_ids] = random_words[random_ids]
        else:
            # 80% of the time, the masked input tokens are replaced with <MASK>
            replaced_ids = torch.bernoulli(torch.full(y.shape, 0.8)).bool() & masked_ids
            x[replaced_ids] = self.tokenizer.mask_token_id

            # 10% of the time, the masked input tokens are replaced with a random word
            random_ids = torch.bernoulli(torch.full(y.shape, 0.5)).bool() & masked_ids & ~replaced_ids
            random_words = torch.multinomial(self.replacement_ids, torch.numel(x), replacement=True).view(x.size())
            x[random_ids] = random_words[random_ids]

        # Replace masked position with float(-inf), True: inf, False: 0
        pad_mask = torch.where(pad_mask, float("-inf"), float(0.0)).type(self.dtype)

        return (global_id, labels, x, y, pad_mask) if self.return_labels else (global_id, x, y, pad_mask)

def DataCollatorLambdaNet(batch):
    """
    Batch is a list of tuples: (global_id, emb, lambdas, flag)

    If emb/lambdas are strings, this returns lists of strings.
    Replace parsing logic as needed.
    """
    global_ids, embs, lambdas, flags = zip(*batch)
    return {
        "global_id": list(global_ids),
        "emb": list(embs),
        "lambdas": list(lambdas),
        "flag": torch.tensor(flags, dtype=torch.long),
    }

class ProteinGymCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[dict]) -> dict:
        # padding
        max_len = max(b["masked_ids"].size(0) for b in batch)
        if max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of # xformer attention layer requires this
        padded_seq = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)

        for j, b in enumerate(batch):
            L = b["masked_ids"].size(0)
            padded_seq[j, :L] = b["masked_ids"]
        return {
            "masked_ids": padded_seq,
            "new_pos": torch.tensor([b["new_pos"]  for b in batch]),
            "dms_idx": [b["dms_idx"] for b in batch],
            "mutants": [b["mutants"] for b in batch],
        }