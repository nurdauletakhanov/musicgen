"""Carve a held-out validation split out of train, per source, in index.json.

The released runs validated (and selected best.pth) on the test split. To keep
the test split untouched, run this once and set `data.val_split: val` in the
training config. Moves whole files (tracks), stratified by source, seeded.

  python -m scripts.make_val_split --chunks ./chunks-44k-1s --per-source 50
"""
import argparse, json, os, random, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default="./chunks-44k-1s")
ap.add_argument("--per-source", type=int, default=50, help="tracks per source to move train -> val")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
idx_p = os.path.join(a.chunks, "index.json"); idx = json.load(open(idx_p))
idx.setdefault("val", {})
os.makedirs(os.path.join(a.chunks, "val"), exist_ok=True)
rng = random.Random(a.seed); moved = {}
for src in sorted({e["source"] for e in idx["train"].values()}):
    keys = sorted(k for k, e in idx["train"].items() if e["source"] == src)
    rng.shuffle(keys)
    for k in keys[:a.per_source]:
        e = idx["train"].pop(k); idx["val"][k] = e
        shutil.move(os.path.join(a.chunks, "train", e["filename"]), os.path.join(a.chunks, "val", e["filename"]))
        moved[src] = moved.get(src, 0) + 1
json.dump(idx, open(idx_p, "w"))
print("moved to val:", moved, "| train left:", len(idx["train"]), "| test untouched:", len(idx["test"]))
