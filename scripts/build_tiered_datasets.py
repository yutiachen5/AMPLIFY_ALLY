"""
Build tiered PLM training data + validation sets from 5 raw sources.

Tiers (train, target 10,000,000 total by default):
  tier1 = PDB (mol:protein, deduped) + UniProt/Swiss-Prot (all) + Pfam-A
          (per-family capped, fills the remainder of the tier1 budget)
  tier2 = UniRef50, pre-filtered to length < 512
          (read directly from data/uniref50_train_max_len_512.csv, the
          name,sequence CSV already produced by fasta_to_csv.py)
  tier3 = MGnify clusters, FL=1 (full-length) preferred, FL=0 backfill if short

At the end, every FASTA output (5 single-source val sets + val_mixed + the 3
tiers) is also converted to CSV via scripts/fasta_to_csv.py, matching the
name,sequence format the AMPLIFY training pipeline expects.

Validation (held out from all tiers above, no train/val leakage):
  - 5 single-source sets: val_pdb, val_uniprot, val_uniref50, val_mgnify, val_pfam
  - 1 mixed set: val_mixed, drawn evenly from the 5 single-source sets

Design notes:
  - Every source that needs both a val and a train split is sampled with ONE
    combined reservoir pass (size = val_size + train_size), then shuffled and
    sliced. This guarantees val/train are disjoint by construction and keeps
    memory bounded to O(val_size + train_size) regardless of source size, so
    the 68GB mgnify.fa.gz / 5.9GB Pfam-A.fasta.gz never need full decompression
    to disk.
  - PDB and UniProt are small enough to fully load, dedup (PDB only), shuffle,
    and slice directly.
  - Pfam additionally needs per-family capping so a handful of huge families
    don't dominate tier1 — done with one reservoir per family, in the same
    single pass as the val_pfam reservoir.
  - A final cross-source leakage filter drops any train record whose exact
    sequence already appears in one of the 5 val sets.

Usage:
  python scripts/build_tiered_datasets.py \
      --pdb data/pdb_seqres.txt \
      --uniprot data/uniprot_sprot.fasta \
      --pfam data/Pfam-A.fasta.gz \
      --uniref50 data/uniref50_train_max_len_512.csv \
      --mgnify data/mgy_clusters.fa.gz \
      --outdir data/tiered
"""

import argparse
import contextlib
import gzip
import io
import random
import zipfile
from collections import defaultdict

from tqdm import tqdm


# --------------------------------------------------------------------------
# FASTA I/O (transparent .gz / .zip / plain, streaming, never fully unzipped
# to disk)
# --------------------------------------------------------------------------

@contextlib.contextmanager
def open_fasta_source(path):
    if path.endswith(".gz"):
        fh = gzip.open(path, "rt")
        try:
            yield fh
        finally:
            fh.close()
    elif path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        try:
            names = zf.namelist()
            name = next((n for n in names if n.lower().endswith((".fasta", ".fa"))), names[0])
            raw = zf.open(name, "r")
            fh = io.TextIOWrapper(raw, encoding="utf-8")
            try:
                yield fh
            finally:
                fh.close()
        finally:
            zf.close()
    else:
        fh = open(path, "rt")
        try:
            yield fh
        finally:
            fh.close()


def fasta_records(path, desc=None):
    header, chunks = None, []
    with open_fasta_source(path) as fh:
        for line in tqdm(fh, desc=desc, unit=" lines", unit_scale=True):
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


def write_fasta(path, records, source_tag):
    with open(path, "w") as f:
        for _uid, header, seq in records:
            f.write(f">{source_tag}|{header}\n{seq}\n")


def fasta_to_csv(input_path, output_path):
    """Same logic as scripts/fasta_to_csv.py, inlined."""
    reader = open(input_path, "r")
    writer = open(output_path, "w")

    writer.write("name,sequence\n")

    name, seq = str(), str()
    for row in tqdm(reader, unit="rows", unit_scale=True):
        if ">" in row:
            if len(name) > 0 and len(seq) > 0:
                writer.write(f"{name},{seq}\n")
                name, seq = str(), str()
            name = row.strip().replace(",", "|")
        else:
            seq += row.strip()
    if len(name) > 0 and len(seq) > 0:
        writer.write(f"{name},{seq}\n")

    reader.close()
    writer.close()


# --------------------------------------------------------------------------
# Reservoir sampling
# --------------------------------------------------------------------------

class Reservoir:
    """Algorithm R: uniform sample of size k from a stream, single pass, O(k) memory."""

    def __init__(self, k, rng):
        self.k = k
        self.rng = rng
        self.items = []
        self.n_seen = 0

    def offer(self, item):
        self.n_seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randint(0, self.n_seen - 1)
            if j < self.k:
                self.items[j] = item


def sample_val_and_train(record_iter, val_size, train_size, rng, desc):
    """One streaming pass -> uniform reservoir of (val_size + train_size),
    shuffled and sliced into disjoint val / train lists."""
    res = Reservoir(val_size + train_size, rng)
    for rec in record_iter:
        res.offer(rec)
    items = res.items
    rng.shuffle(items)
    val = items[:val_size]
    train = items[val_size:val_size + train_size]
    if len(items) < val_size + train_size:
        print(f"[{desc}] WARNING: pool only had {res.n_seen} filtered records, "
              f"wanted {val_size + train_size} (val={len(val)}, train={len(train)})")
    else:
        print(f"[{desc}] pool={res.n_seen} -> val={len(val)}, train={len(train)}")
    return val, train


# --------------------------------------------------------------------------
# Source-specific filters
# --------------------------------------------------------------------------

def iter_pdb_protein_deduped(path):
    """mol:protein only, dedup identical chains (e.g. homomultimers)."""
    seen = set()
    n_raw, n_dup, n_na = 0, 0, 0
    for header, seq in fasta_records(path, desc="pdb"):
        if "mol:protein" not in header:
            n_na += 1
            continue
        n_raw += 1
        h = hash(seq)
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        uid = header.split()[0]
        yield uid, header, seq
    print(f"[pdb] mol:na skipped={n_na}, mol:protein seen={n_raw}, exact-duplicate chains skipped={n_dup}")


def iter_uniprot(path):
    for header, seq in fasta_records(path, desc="uniprot"):
        yield header.split()[0], header, seq


def iter_uniref50_from_csv(path):
    """Read the pre-filtered (length < 512) name,sequence CSV produced by
    fasta_to_csv.py directly — no re-filtering needed. The `name` column
    retains fasta_to_csv.py's leading '>' from the original header line."""
    n = 0
    with open(path) as f:
        f.readline()  # header: "name,sequence"
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, seq = line.split(",", 1)
            header = name[1:] if name.startswith(">") else name
            n += 1
            yield header.split()[0], header, seq
    print(f"[uniref50] loaded {n} pre-filtered (<512aa) records from {path}")


def pfam_family(header):
    fields = header.split()
    if len(fields) < 3:
        return "UNKNOWN"
    return fields[2].split(".")[0]


def iter_mgnify_filtered(path, full_length_only):
    n_total, n_kept = 0, 0
    for header, seq in fasta_records(path, desc="mgnify"):
        n_total += 1
        fields = header.split()
        fl = fields[1] if len(fields) > 1 else ""
        if full_length_only and fl != "FL=1":
            continue
        n_kept += 1
        yield fields[0], header, seq
    tag = "FL=1" if full_length_only else "any"
    print(f"[mgnify {tag}] total scanned={n_total}, kept={n_kept}")


# --------------------------------------------------------------------------
# Pfam: single pass building (a) a uniform val reservoir and (b) per-family
# capped train reservoirs, at the same time.
# --------------------------------------------------------------------------

def sample_pfam(path, val_size, family_cap, pfam_target, rng):
    val_res = Reservoir(val_size, rng)
    family_res = defaultdict(lambda: Reservoir(family_cap, rng))
    n_total = 0
    for header, seq in fasta_records(path, desc="pfam"):
        n_total += 1
        rec = (header.split()[0], header, seq)
        val_res.offer(rec)
        family_res[pfam_family(header)].offer(rec)
    val_pfam = val_res.items
    rng.shuffle(val_pfam)
    val_uids = {r[0] for r in val_pfam}

    candidates = [rec for res in family_res.values() for rec in res.items if rec[0] not in val_uids]
    rng.shuffle(candidates)
    print(f"[pfam] total={n_total}, families={len(family_res)}, "
          f"family-capped candidates={len(candidates)}, target={pfam_target}")
    if len(candidates) < pfam_target:
        print(f"[pfam] WARNING: only {len(candidates)} candidates after family cap "
              f"({family_cap}/family) and val exclusion, short of target {pfam_target}. "
              f"Raise --pfam-family-cap to fill the gap.")
        pfam_train = candidates
    else:
        pfam_train = candidates[:pfam_target]
    return val_pfam, pfam_train


# --------------------------------------------------------------------------
# Cross-source leakage guard: drop any train record whose exact sequence
# already appears in one of the 5 held-out val sets.
# --------------------------------------------------------------------------

def filter_leakage(train_records, val_seq_hashes, desc):
    kept = [r for r in train_records if hash(r[2]) not in val_seq_hashes]
    dropped = len(train_records) - len(kept)
    if dropped:
        print(f"[{desc}] dropped {dropped} record(s) matching a val sequence exactly (leakage guard)")
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", default="data/pdb_seqres.txt")
    p.add_argument("--uniprot", default="data/uniprot_sprot.fasta")
    p.add_argument("--pfam", default="data/Pfam-A.fasta.gz")
    p.add_argument("--uniref50", default="data/uniref50_train_max_len_512.csv")
    p.add_argument("--mgnify", default="data/mgy_clusters.fa.gz")
    p.add_argument("--outdir", default="data/tiered")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--val-per-source", type=int, default=10_000,
                    help="size of each of the 5 single-source val sets")
    p.add_argument("--val-mixed-total", type=int, default=10_000,
                    help="total size of the mixed val set, split evenly across the 5 sources")

    p.add_argument("--tier1-size", type=int, default=4_000_000)
    p.add_argument("--tier2-size", type=int, default=3_500_000)
    p.add_argument("--tier3-size", type=int, default=2_500_000)

    p.add_argument("--uniref-expected-pool", type=int, default=54_243_229,
                    help="sanity-check the observed length-filtered pool against this")
    p.add_argument("--pfam-family-cap", type=int, default=300,
                    help="max sequences kept per Pfam family before trimming to tier1's fill target")

    args = p.parse_args()
    rng = random.Random(args.seed)

    import os
    os.makedirs(f"{args.outdir}/val", exist_ok=True)
    os.makedirs(f"{args.outdir}/train", exist_ok=True)

    # --- PDB (small: fully load, dedup, shuffle, slice) ----------------------
    pdb_all = list(iter_pdb_protein_deduped(args.pdb))
    rng.shuffle(pdb_all)
    val_pdb = pdb_all[:args.val_per_source]
    pdb_train = pdb_all[args.val_per_source:]
    print(f"[pdb] deduped pool={len(pdb_all)} -> val={len(val_pdb)}, train={len(pdb_train)}")

    # --- UniProt (small: fully load, shuffle, slice) -------------------------
    uniprot_all = list(iter_uniprot(args.uniprot))
    rng.shuffle(uniprot_all)
    val_uniprot = uniprot_all[:args.val_per_source]
    uniprot_train = uniprot_all[args.val_per_source:]
    print(f"[uniprot] pool={len(uniprot_all)} -> val={len(val_uniprot)}, train={len(uniprot_train)}")

    # --- Pfam: fill whatever's left of tier1 after PDB + UniProt -------------
    pfam_target = args.tier1_size - len(pdb_train) - len(uniprot_train)
    if pfam_target <= 0:
        raise SystemExit(f"tier1-size ({args.tier1_size}) is smaller than PDB+UniProt "
                          f"training pools ({len(pdb_train) + len(uniprot_train)}); increase --tier1-size")
    val_pfam, pfam_train = sample_pfam(args.pfam, args.val_per_source, args.pfam_family_cap, pfam_target, rng)

    # --- UniRef50 (length filter, combined val+train reservoir) --------------
    val_uniref50, uniref50_train = sample_val_and_train(
        iter_uniref50_from_csv(args.uniref50),
        args.val_per_source, args.tier2_size, rng, desc="uniref50",
    )
    observed_pool = len(val_uniref50) + len(uniref50_train)
    if observed_pool < args.tier2_size and abs(observed_pool - args.uniref_expected_pool) > 0.1 * args.uniref_expected_pool:
        print(f"[uniref50] WARNING: pool ({observed_pool}+) looks far from the expected "
              f"~{args.uniref_expected_pool} — check {args.uniref50}")

    # --- MGnify (FL=1 preferred, combined val+train reservoir, FL=0 backfill)
    val_mgnify, mgnify_train = sample_val_and_train(
        iter_mgnify_filtered(args.mgnify, full_length_only=True),
        args.val_per_source, args.tier3_size, rng, desc="mgnify(FL=1)",
    )
    shortfall = (args.val_per_source + args.tier3_size) - (len(val_mgnify) + len(mgnify_train))
    if shortfall > 0:
        print(f"[mgnify] FL=1 pool short by {shortfall}, backfilling from FL=0 fragments")
        chosen_uids = {r[0] for r in val_mgnify} | {r[0] for r in mgnify_train}
        backfill_pool = [r for r in iter_mgnify_filtered(args.mgnify, full_length_only=False)
                         if r[0] not in chosen_uids]
        rng.shuffle(backfill_pool)
        backfill = backfill_pool[:shortfall]
        need_val = args.val_per_source - len(val_mgnify)
        val_mgnify += backfill[:need_val]
        mgnify_train += backfill[need_val:]

    # --- Mixed val: split val_mixed_total evenly across the 5 val sets -------
    per_source_pools = {
        "PDB": val_pdb, "UNIPROT": val_uniprot, "PFAM": val_pfam,
        "UNIREF50": val_uniref50, "MGNIFY": val_mgnify,
    }
    n_sources = len(per_source_pools)
    base = args.val_mixed_total // n_sources
    remainder = args.val_mixed_total - base * n_sources
    val_mixed_tagged = []
    for i, (tag, pool) in enumerate(per_source_pools.items()):
        take = base + (1 if i < remainder else 0)
        take = min(take, len(pool))
        picked = rng.sample(pool, take)
        val_mixed_tagged.extend((tag, r) for r in picked)
    print(f"[val_mixed] total={len(val_mixed_tagged)} (target {args.val_mixed_total})")

    # --- Cross-source leakage guard: drop train records that exactly match a
    # held-out val sequence from ANY of the 5 sources ------------------------
    val_seq_hashes = set()
    for pool in per_source_pools.values():
        val_seq_hashes.update(hash(r[2]) for r in pool)

    pdb_train = filter_leakage(pdb_train, val_seq_hashes, "pdb")
    uniprot_train = filter_leakage(uniprot_train, val_seq_hashes, "uniprot")
    pfam_train = filter_leakage(pfam_train, val_seq_hashes, "pfam")
    uniref50_train = filter_leakage(uniref50_train, val_seq_hashes, "uniref50")
    mgnify_train = filter_leakage(mgnify_train, val_seq_hashes, "mgnify")

    # --- Write val outputs -----------------------------------------------------
    write_fasta(f"{args.outdir}/val/val_pdb.fasta", val_pdb, "PDB")
    write_fasta(f"{args.outdir}/val/val_uniprot.fasta", val_uniprot, "UNIPROT")
    write_fasta(f"{args.outdir}/val/val_pfam.fasta", val_pfam, "PFAM")
    write_fasta(f"{args.outdir}/val/val_uniref50.fasta", val_uniref50, "UNIREF50")
    write_fasta(f"{args.outdir}/val/val_mgnify.fasta", val_mgnify, "MGNIFY")
    with open(f"{args.outdir}/val/val_mixed.fasta", "w") as f:
        for tag, (uid, header, seq) in val_mixed_tagged:
            f.write(f">{tag}|{header}\n{seq}\n")

    # --- Write tier outputs (shuffled within each tier) -------------------------
    rng.shuffle(uniref50_train)
    rng.shuffle(mgnify_train)

    tier1_pairs = list(zip(
        ["UNIPROT"] * len(uniprot_train) + ["PDB"] * len(pdb_train) + ["PFAM"] * len(pfam_train),
        uniprot_train + pdb_train + pfam_train,
    ))
    rng.shuffle(tier1_pairs)
    with open(f"{args.outdir}/train/tier1.fasta", "w") as f:
        for tag, (uid, header, seq) in tier1_pairs:
            f.write(f">{tag}|{header}\n{seq}\n")

    write_fasta(f"{args.outdir}/train/tier2.fasta", uniref50_train, "UNIREF50")
    write_fasta(f"{args.outdir}/train/tier3.fasta", mgnify_train, "MGNIFY")

    # --- Manifest ---------------------------------------------------------------
    manifest_path = f"{args.outdir}/manifest.csv"
    with open(manifest_path, "w") as f:
        f.write("split,source,count\n")
        f.write(f"val,pdb,{len(val_pdb)}\n")
        f.write(f"val,uniprot,{len(val_uniprot)}\n")
        f.write(f"val,pfam,{len(val_pfam)}\n")
        f.write(f"val,uniref50,{len(val_uniref50)}\n")
        f.write(f"val,mgnify,{len(val_mgnify)}\n")
        f.write(f"val,mixed,{len(val_mixed_tagged)}\n")
        f.write(f"tier1,uniprot,{len(uniprot_train)}\n")
        f.write(f"tier1,pdb,{len(pdb_train)}\n")
        f.write(f"tier1,pfam,{len(pfam_train)}\n")
        f.write(f"tier1,total,{len(tier1_pairs)}\n")
        f.write(f"tier2,uniref50,{len(uniref50_train)}\n")
        f.write(f"tier3,mgnify,{len(mgnify_train)}\n")
        total_train = len(tier1_pairs) + len(uniref50_train) + len(mgnify_train)
        f.write(f"train,total,{total_train}\n")
    print(f"Manifest written to {manifest_path}. Total train samples: {total_train}")

    # --- Convert every FASTA output to CSV (same logic as fasta_to_csv.py) -----
    fasta_outputs = [
        f"{args.outdir}/val/val_pdb.fasta",
        f"{args.outdir}/val/val_uniprot.fasta",
        f"{args.outdir}/val/val_pfam.fasta",
        f"{args.outdir}/val/val_uniref50.fasta",
        f"{args.outdir}/val/val_mgnify.fasta",
        f"{args.outdir}/val/val_mixed.fasta",
        f"{args.outdir}/train/tier1.fasta",
        f"{args.outdir}/train/tier2.fasta",
        f"{args.outdir}/train/tier3.fasta",
    ]
    for fasta_path in fasta_outputs:
        csv_path = fasta_path[: -len(".fasta")] + ".csv"
        fasta_to_csv(fasta_path, csv_path)
        print(f"[fasta_to_csv] {fasta_path} -> {csv_path}")


if __name__ == "__main__":
    main()
