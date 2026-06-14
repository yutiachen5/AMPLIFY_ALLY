# adapted from: https://github.com/chandar-lab/AMPLIFY/blob/main/examples/contact_prediction.ipynb

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import sys
sys.path.append("/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/examples")
from utils import load_pickle_dataset, load_from_hf, load_from_mila, apc, symmetrize


def get_attn_map(model, tokenizer, protein, device, fp16):
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=fp16):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        x = torch.as_tensor(tokenizer.encode(protein)).to(torch.long)  # tokenize the protein
        x = x.unsqueeze(0).to(device)
        attn_map = model(x, output_attentions=True)["attentions"]
        attn_map = torch.stack(attn_map).detach().cpu()  # stack the attention maps and move to CPU
        attn_map = attn_map.reshape(-1, x.size(-1), x.size(-1))  # (map, residues, residues)
        attn_map = attn_map[:, 1:-1, 1:-1]  # remove special tokens <bos> and <eos>
        attn_map = apc(symmetrize(attn_map))  # process the attention maps
        attn_map = attn_map.permute(1, 2, 0)  # (residues, residues, map)
        return attn_map


def compute_jacobian(model, tokenizer, protein, device, fp16, batch_size=32):

    # Get the IDs of the amino acids
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    amino_acids_ids = tokenizer.encode(amino_acids, add_special_tokens=False)

    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=fp16):
        # Tokenize the sequence
        input = torch.as_tensor(tokenizer.encode(protein)).to(torch.long)  # tokenize the protein
        length = len(protein)

        # For each position in the sequence, prepare the input with all mutations
        mutated_inputs = []
        for n in range(len(protein)):
            x = torch.tile(input, [20, 1])
            x[:, n] = torch.as_tensor(amino_acids_ids)
            mutated_inputs.append(x)
        mutated_inputs = torch.cat(mutated_inputs, dim=0)

        # Get the model's logits without mutations
        ref_logits = model(input.unsqueeze(0).to(device))["logits"].squeeze()

        # Remove the special tokens and keep only the logits for amino acids
        ref_logits = ref_logits[..., 1:-1, amino_acids_ids].cpu().numpy().astype(np.float64)

        # Compute the logits for all mutations
        mutated_logits = []
        for batch in torch.split(mutated_inputs, batch_size):
            mutated_logits.append(model(batch.to(device))["logits"][..., 1:-1, amino_acids_ids])
        mutated_logits = (
            torch.cat(mutated_logits, dim=0).reshape(length, 20, length, 20).cpu().numpy().astype(np.float64)
        )

    # Compute the jacobian
    jac = mutated_logits - ref_logits

    # Symmetrize the jacobian
    jac = (jac + jac.transpose(2, 3, 0, 1)) / 2

    # Center the jacobian
    for i in range(4):
        jac -= jac.mean(i, keepdims=True)

    # Collapse (L, 20, L, 20) -> (L, L)
    jac = np.sqrt(np.square(jac).sum((1, 3)))

    # Remove diagonal (contacts with itself)
    np.fill_diagonal(jac, 0)

    # APC
    a1 = jac.sum(0, keepdims=True)
    a2 = jac.sum(1, keepdims=True)
    jac = jac - (a1 * a2) / jac.sum()

    # Remove diagonal (contacts with itself)
    np.fill_diagonal(jac, 0)

    return torch.tensor(jac).unsqueeze(-1)


# Adapted from https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: torch.Tensor = None,
    minsep: int = 6,
    maxsep: int = None,
    override_length: int = None,
):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    override_length = (targets[0, 0] >= 0).sum()

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, " f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(gather_lengths, device=device)

    gather_indices = (torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(binned_cumulative_dist)

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}


def evaluate_prediction(predictions: torch.Tensor, targets: torch.Tensor):
    contact_ranges = [("local", 3, 6), ("short", 6, 12), ("medium", 12, 24), ("long", 24, None)]
    metrics = {}
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(predictions, targets, minsep=minsep, maxsep=maxsep)
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val.item()
    return metrics


# Adapted from https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
def plot_contacts_and_predictions(predictions, contacts, ax, cmap="Blues", ms=1, title_text=True) -> None:

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(contacts, torch.Tensor):
        contacts = contacts.detach().cpu().numpy()
    if ax is None:
        ax = plt.gca()

    seqlen = contacts.shape[0]
    relative_distance = np.add.outer(-np.arange(seqlen), np.arange(seqlen))
    bottom_mask = relative_distance < 0
    masked_image = np.ma.masked_where(bottom_mask, predictions)
    invalid_mask = np.abs(np.add.outer(np.arange(seqlen), -np.arange(seqlen))) < 6
    predictions = predictions.copy()
    predictions[invalid_mask] = float("-inf")

    topl_val = np.sort(predictions.reshape(-1))[-seqlen]
    pred_contacts = predictions >= topl_val
    true_positives = contacts & pred_contacts & ~bottom_mask
    false_positives = ~contacts & pred_contacts & ~bottom_mask
    other_contacts = contacts & ~pred_contacts & ~bottom_mask

    img = ax.imshow(masked_image, cmap=cmap)
    oc = ax.plot(*np.where(other_contacts), "o", c="grey", ms=ms)[0]
    fn = ax.plot(*np.where(false_positives), "o", c="r", ms=ms)[0]
    tp = ax.plot(*np.where(true_positives), "o", c="b", ms=ms)[0]
    ti = ax.set_title(title_text) if title_text is not None else None

    ax.axis("square")
    ax.set_xlim([0, seqlen])
    ax.set_ylim([0, seqlen])


def main(args):
    all_scores = []

    # Load model
    if args.source == "hf":
        model, tokenizer = load_from_hf(args.model_path, args.tokenizer_path, fp16=args.fp16)
    elif args.source == "mila":
        model, tokenizer = load_from_mila(args.model_path, args.config_path)
    model.to(args.device)
    model = torch.compile(model, disable=not args.compile)

    # Load dataset
    labels, proteins, dist_matrices = load_pickle_dataset(args.data_path, args.n_proteins, args.max_length)

    # Random split
    labels_train, labels_test, proteins_train, proteins_test, dist_matrices_train, dist_matrices_test = train_test_split(
        labels, proteins, dist_matrices, train_size=args.n_train_samples, random_state=args.seed
    )

    # Compute contacts maps from distance matrices
    contact_maps_train = list(map(lambda x: x < args.threshold_c_alpha, dist_matrices_train))
    contact_maps_test  = list(map(lambda x: x < args.threshold_c_alpha, dist_matrices_test))

    # Only the Attention method of ESM requires training a logistic regression model to weight the attention maps
    if args.method == "Attention":
        X_train, y_train = list(), list()
        # Create a lower diagonal mask with an offset of min_sep_eval and flatten
        for protein, contact_map in zip(proteins_train, contact_maps_train):
            pos = np.arange(contact_map.shape[0])
            if args.method == "Attention":
                attn_map = get_attn_map(model, tokenizer, protein, args.device, args.fp16)
            elif args.method == "Jacobian":
                attn_map = compute_jacobian(model, tokenizer, protein, args.device, args.fp16, batch_size=args.batch_size)
            diag_idx = np.expand_dims(pos, axis=0) - np.expand_dims(pos, axis=1) >= args.min_sep
            X_train.extend(attn_map[diag_idx, :].reshape(-1, attn_map.shape[-1]).to(torch.float32))
            y_train.extend(contact_map[diag_idx].reshape(-1))
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)

        # Logistic Regression (careful, liblinear does not support int64!)
        clf = LogisticRegression(solver="liblinear", penalty="l1", C=args.l1_penalty)
        clf.fit(X_train, y_train)

    # Predict and evaluate
    for i, (label, protein, y_true) in enumerate(zip(labels_test, proteins_test, contact_maps_test)):
        if args.method == "Attention":
            x = get_attn_map(model, tokenizer, protein, args.device, args.fp16)
            y_pred = clf.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(y_true.shape)
        elif args.method == "Jacobian":
            y_pred = compute_jacobian(model, tokenizer, protein, args.device, args.fp16, batch_size=args.batch_size).reshape(y_true.shape)

        scores = evaluate_prediction(y_pred, y_true)
        scores["label"] = label
        all_scores.append(scores)

        # Visualize
        if i % 3 == 0:
            fig, axes = plt.subplots(figsize=(18, 6), ncols=3)

        plot_contacts_and_predictions(
            y_pred,
            y_true,
            ax=axes[i % 3],
            title_text=f"{label}: Long Range P@L: {scores['long_P@L']:0.1%}",
        )

        if i % 3 == 2:
            plt.show()
            plt.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.o)), exist_ok=True)
    plt.savefig(os.path.join(args.o, f"{args.p}.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    df = pd.DataFrame(all_scores)
    df.to_csv(os.path.join(args.o, f"{args.p}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact map prediction")

    # Model
    parser.add_argument("--source", type=str, default="mila", choices=["mila", "hf"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None) # this is for hf model only
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--method", type=str, default="Jacobian", choices=["Attention", "Jacobian"])

    # Dataset
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_proteins", type=int, default=25)
    parser.add_argument("--max_length", type=int, default=512)

    # Logistic Regression
    parser.add_argument("--n_train_samples", type=int,   default=20)
    parser.add_argument("--threshold_c_alpha", type=float, default=8.0)
    parser.add_argument("--min_sep", type=int,   default=6)
    parser.add_argument("--l1_penalty", type=float, default=0.15)
    parser.add_argument("--seed", type=int,   default=0)

    # Output
    parser.add_argument("-o", type=str, default="/hpc/group/naderilab/eleanor/AMPLIFY_ALLY/outputs/")
    parser.add_argument("-p", type=str, default="AMPLIFY_Contact_CASP15")
    args = parser.parse_args()

    main(args)