"""
Check two name,sequence CSV files (the format produced by fasta_to_csv.py and read
by InMemoryProteinDataset) for exact-sequence overlap and, if any is found, write a
copy of the train CSV with the overlapping sequences removed.

Overlap is checked by exact sequence match (same convention as the leakage guard in
build_tiered_datasets.py) -- it will not catch near-duplicates/near-identical
sequences, only byte-for-byte identical ones.

Usage:
    python scripts/remove_train_val_overlap.py \
        --train data/tiered/train/tier2.csv \
        --other data/tiered/val/val_uniref50.csv \
        --out data/tiered/train/tier2.deduped.csv
"""

from argparse import ArgumentParser

from tqdm import tqdm


def csv_records(path):
    """Yield (name, sequence) tuples from a name,sequence CSV, skipping the header."""
    with open(path, "r") as fh:
        next(fh)  # skip header
        for line in tqdm(fh, desc=f"reading {path}", unit=" rows", unit_scale=True):
            line = line.rstrip("\n")
            if not line:
                continue
            name, seq = line.split(",", 1)
            yield name, seq


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="CSV file to check/clean (the train set).")
    parser.add_argument("--other", required=True, help="CSV file to check against (e.g. a val set or candidate pool).")
    parser.add_argument("--out", required=True, help="Path to write the deduped train CSV to.")
    args = parser.parse_args()

    other_hashes = {hash(seq) for _, seq in csv_records(args.other)}
    print(f"[other] {len(other_hashes)} unique sequences loaded from {args.other}")

    n_total, n_overlap = 0, 0
    with open(args.out, "w") as out_f:
        out_f.write("name,sequence\n")
        for name, seq in csv_records(args.train):
            n_total += 1
            if hash(seq) in other_hashes:
                n_overlap += 1
                continue
            out_f.write(f"{name},{seq}\n")

    print(f"[train] {n_total} total sequences, {n_overlap} overlapping with {args.other} removed")
    print(f"[train] {n_total - n_overlap} sequences written to {args.out}")


if __name__ == "__main__":
    main()
