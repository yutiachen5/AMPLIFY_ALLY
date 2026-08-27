"""
Check two FASTA files for exact-sequence overlap and, if any is found, write a copy
of the train FASTA with the overlapping sequences removed.

Overlap is checked by exact sequence match (same convention as the leakage guard in
build_tiered_datasets.py) -- it will not catch near-duplicates/near-identical
sequences, only byte-for-byte identical ones.

Usage:
    python scripts/remove_train_val_overlap.py \
        --train data/tiered/train/tier2.fasta \
        --other data/tiered/val/val_uniref50.fasta \
        --out data/tiered/train/tier2.deduped.fasta
"""

from argparse import ArgumentParser

from tqdm import tqdm


def fasta_records(path):
    """Yield (header, sequence) tuples, joining multi-line sequences."""
    header, chunks = None, []
    with open(path, "r") as fh:
        for line in tqdm(fh, desc=f"reading {path}", unit=" lines", unit_scale=True):
            line = line.rstrip("\n")
            if not line:
                continue
            if line[0] == ">":
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="FASTA file to check/clean (the train set).")
    parser.add_argument("--other", required=True, help="FASTA file to check against (e.g. a val set or candidate pool).")
    parser.add_argument("--out", required=True, help="Path to write the deduped train FASTA to.")
    args = parser.parse_args()

    other_hashes = {hash(seq) for _, seq in fasta_records(args.other)}
    print(f"[other] {len(other_hashes)} unique sequences loaded from {args.other}")

    n_total, n_overlap = 0, 0
    with open(args.out, "w") as out_f:
        for header, seq in fasta_records(args.train):
            n_total += 1
            if hash(seq) in other_hashes:
                n_overlap += 1
                continue
            out_f.write(f">{header}\n{seq}\n")

    print(f"[train] {n_total} total sequences, {n_overlap} overlapping with {args.other} removed")
    print(f"[train] {n_total - n_overlap} sequences written to {args.out}")


if __name__ == "__main__":
    main()
