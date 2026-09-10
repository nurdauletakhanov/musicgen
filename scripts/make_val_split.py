"""Carve a held-out validation split out of train, per source, in index.json.

The released runs validated (and selected best.pth) on the test split. To keep
the test split untouched, run this once and set `data.val_split: val` in the
training config. Moves whole files (tracks), stratified by source, seeded.

  python -m scripts.make_val_split --chunks ./chunks-44k-1s --per-source 50
"""
import argparse, json, os, random, shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", default="./chunks-44k-1s")
    ap.add_argument("--per-source", type=int, default=50,
                    help="tracks per source to move train -> val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="carve again even if a val split already exists")
    a = ap.parse_args()

    idx_p = os.path.join(a.chunks, "index.json")
    idx = json.load(open(idx_p))

    # Running this twice takes a SECOND subset out of train. Refuse unless asked.
    if idx.get("val") and not a.force:
        raise SystemExit(
            f"{idx_p} already has a val split of {len(idx['val'])} files. "
            "Re-running would carve another subset out of train; pass --force "
            "if that is really what you want.")

    idx.setdefault("val", {})
    os.makedirs(os.path.join(a.chunks, "val"), exist_ok=True)
    rng = random.Random(a.seed)
    moved = {}
    manifest = []
    for src in sorted({e["source"] for e in idx["train"].values()}):
        keys = sorted(k for k, e in idx["train"].items() if e["source"] == src)
        rng.shuffle(keys)
        for k in keys[:a.per_source]:
            e = idx["train"][k]
            dst = os.path.join(a.chunks, "val", e["filename"])
            src_p = os.path.join(a.chunks, "train", e["filename"])
            # Resumable: a file already moved by an interrupted run is fine.
            if os.path.exists(src_p):
                shutil.move(src_p, dst)
            elif not os.path.exists(dst):
                print(f"  [warn] {k}: file missing from both splits, skipping")
                continue
            idx["val"][k] = idx["train"].pop(k)
            moved[src] = moved.get(src, 0) + 1
            manifest.append({"key": k, "source": src, "filename": e["filename"]})

    # Write the index only after every move, and record what was moved so the
    # split is auditable and re-derivable.
    with open(idx_p, "w") as fh:
        json.dump(idx, fh)
    with open(os.path.join(a.chunks, "val_split_manifest.json"), "w") as fh:
        json.dump({"seed": a.seed, "per_source": a.per_source,
                   "n_moved": len(manifest), "tracks": manifest}, fh, indent=2)
    print("moved to val:", moved, "| train left:", len(idx["train"]),
          "| test untouched:", len(idx["test"]))
    print("wrote", os.path.join(a.chunks, "val_split_manifest.json"))


if __name__ == "__main__":
    main()
