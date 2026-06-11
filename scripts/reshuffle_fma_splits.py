"""
Move FMA tracks from the unified chunks dir's `train/` into `test/`, so eval
isn't MAESTRO-only.

Two modes, auto-selected by whether `fma_metadata/tracks.csv` is on disk:

  1) Official split (preferred): read `tracks.csv`, move every track whose
     `('set', 'split') == 'test'` from train/ to test/. FMA ships an 80/10/10
     split so this is ~10,600 test tracks. Tracks marked `'validation'` stay
     in train (we don't do HP search, so validation serves no separate role).

  2) Random fallback: seeded 5% sample of FMA tracks in train. Same interface.

Operates purely on filesystem + index.json (no audio re-decoding). Reversible
(just run with --reverse later, or move files back manually).

    # after downloading fma_metadata.zip and extracting to dataset/fma_metadata/
    python -m scripts.reshuffle_fma_splits
"""

import argparse
import csv
import json
import os
import random
import shutil
from typing import Optional, Set


def _load_official_fma_test_ids(meta_csv: str) -> Set[int]:
    """Return set of FMA track IDs whose official split is 'test'.

    FMA's `tracks.csv` is a multi-index CSV: first column is the track_id
    (integer), and the `split` column lives under the `('set', 'split')`
    multi-index header. Headers span rows 0-2; data starts at row 3.
    """
    ids: Set[int] = set()
    with open(meta_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        # Row 0 groups the multi-index top ("set","set","track","track",...)
        row0 = next(reader)
        # Row 1 is the leaf name ("subset","split","title","artist",...)
        row1 = next(reader)
        # Row 2 is the dtype hint ("", "", "", "", ...); skip it
        next(reader)
        # Find the split column — the leaf label 'split' under group 'set'.
        split_col = None
        for i, (top, leaf) in enumerate(zip(row0, row1)):
            if top == "set" and leaf == "split":
                split_col = i
                break
        if split_col is None:
            raise RuntimeError("tracks.csv: could not locate ('set','split') column")
        for row in reader:
            if len(row) <= split_col:
                continue
            track_id_s = row[0].strip()
            split_s = row[split_col].strip().lower()
            if split_s == "test" and track_id_s.isdigit():
                ids.add(int(track_id_s))
    return ids


def _tally(d):
    from collections import Counter
    tc, cc = Counter(), Counter()
    for e in d.values():
        s = e.get("source", "unknown")
        tc[s] += 1
        cc[s] += e.get("num_chunks", 0)
    return tc, cc


def _fma_key_to_track_id(key: str) -> Optional[int]:
    # key format: "fma__000002"
    if not key.startswith("fma__"):
        return None
    tail = key.split("__", 1)[1]
    return int(tail) if tail.isdigit() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks-dir", type=str, default="./chunks-44k-1s")
    p.add_argument("--fma-metadata-csv", type=str,
                   default="./dataset/fma_metadata/tracks.csv",
                   help="Path to FMA's tracks.csv. If missing, falls back to random split.")
    p.add_argument("--random-fraction", type=float, default=0.05,
                   help="Used only if metadata CSV isn't found")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    index_path = os.path.join(args.chunks_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    train = index.setdefault("train", {})
    test = index.setdefault("test", {})

    fma_train_keys = sorted(k for k, e in train.items() if e.get("source") == "fma")

    # Decide source of truth for which tracks go to test.
    selected: Set[str]
    if os.path.isfile(args.fma_metadata_csv):
        print(f"[mode] official splits from {args.fma_metadata_csv}")
        official_test_ids = _load_official_fma_test_ids(args.fma_metadata_csv)
        print(f"       {len(official_test_ids):,} tracks marked 'test' in tracks.csv")
        selected = set()
        for k in fma_train_keys:
            tid = _fma_key_to_track_id(k)
            if tid is not None and tid in official_test_ids:
                selected.add(k)
    else:
        print(f"[mode] random {args.random_fraction*100:.1f}% (seed={args.seed}) "
              f"— metadata not found at {args.fma_metadata_csv}")
        n_move = int(round(len(fma_train_keys) * args.random_fraction))
        rng = random.Random(args.seed)
        selected = set(rng.sample(fma_train_keys, n_move))

    moved_chunks = sum(train[k]["num_chunks"] for k in selected)
    print(f"plan: move {len(selected):,} / {len(fma_train_keys):,} FMA tracks "
          f"from train -> test ({moved_chunks:,} chunks)")

    if not selected:
        print("nothing to move.")
        return
    if args.dry_run:
        print("[dry-run] no changes made")
        return

    train_dir = os.path.join(args.chunks_dir, "train")
    test_dir = os.path.join(args.chunks_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    n_done, n_fail = 0, 0
    for key in sorted(selected):
        entry = train[key]
        fn = entry["filename"]
        src = os.path.join(train_dir, fn)
        dst = os.path.join(test_dir, fn)
        if not os.path.exists(src):
            print(f"  MISSING: {key} ({src})")
            n_fail += 1
            continue
        try:
            shutil.move(src, dst)
        except OSError as e:
            print(f"  move failed {key}: {e}")
            n_fail += 1
            continue
        test[key] = entry
        del train[key]
        n_done += 1

    tmp = index_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, index_path)

    t_tr, c_tr = _tally(train)
    t_te, c_te = _tally(test)
    print(f"\ndone. moved {n_done:,} tracks, failed {n_fail}.")
    print("train:")
    for s in sorted(t_tr):
        print(f"  {s}: {t_tr[s]:,} tracks, {c_tr[s]:,} chunks")
    print("test:")
    for s in sorted(t_te):
        print(f"  {s}: {t_te[s]:,} tracks, {c_te[s]:,} chunks")


if __name__ == "__main__":
    main()
