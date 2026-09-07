"""Paired bootstrap confidence intervals for the stem-subtraction results.

The subtraction eval measures every model on the *same* (track, chunk, stem)
units: chunk selection is a pure function of the seed and the sorted track
list, so measurements align one-to-one across models. That makes a paired
analysis valid and much tighter than comparing two independent means, because
"some chunks are simply harder" cancels out.

For each model we report a percentile bootstrap CI on the mean SI-SDR, and for
each contrast (a model against its matched control) a CI on the *paired mean
difference*, plus the fraction of units that improved.

What this does and does not cover: it quantifies evaluation variance (would a
different draw of test audio change the conclusion?). It does not quantify
training variance (would a different random initialization change it?) -- only
independent re-training answers that.

Usage:
  python -m evaluation.paired_stats \
      --per-chunk evaluation/v2_metrics/per_chunk/*.json \
      --out evaluation/v2_metrics/paired_stats.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

# (treatment, control, label) -- each pair differs in exactly one factor.
CONTRASTS = [
    ("v2.1-decmix", "v2.0-continued", "L_dec vs no mix (7.66x)"),
    ("v2.2-decmix-disc", "v2.0-continued", "L_dec+disc vs no mix (7.66x)"),
    ("v2.2-decmix-disc", "v2.1-decmix", "disc-on-mix vs L_dec alone"),
    ("v3.1-decmix-disc-d64", "v3.0-baseline-d64", "mix vs no mix (15.3x)"),
]
KEY = "sub"          # metric of record: latent subtraction vs ground truth
N_BOOT = 10000
SEED = 0


def load(paths):
    """run name -> {(track, stem, chunk): value} plus the ordered key list."""
    models = {}
    for p in paths:
        name = re.sub(r"_per_chunk\.json$", "", os.path.basename(p))
        d = json.load(open(p))
        models[name] = {(r["track"], r["stem"], r["chunk"]): r for r in d["records"]}
        print(f"  {name}: {len(models[name])} measurements")
    return models


def ci(vals, rng, n_boot=N_BOOT):
    """Percentile bootstrap CI for the mean."""
    vals = np.asarray(vals, dtype=np.float64)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-chunk", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    paths = [p for pat in a.per_chunk for p in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no per-chunk files matched: {a.per_chunk}")
    print("loading:")
    models = load(paths)
    rng = np.random.default_rng(SEED)

    out = {"metric": KEY, "n_boot": a.n_boot, "models": {}, "contrasts": {}}

    for name, recs in sorted(models.items()):
        vals = [r[KEY] for r in recs.values()]
        m, lo, hi = ci(vals, rng, a.n_boot)
        out["models"][name] = {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}
        print(f"{name:26s} {m:+.2f} dB  95% CI [{lo:+.2f}, {hi:+.2f}]  n={len(vals)}")

    print("\npaired contrasts (same chunks, same stems):")
    for treat, ctrl, label in CONTRASTS:
        if treat not in models or ctrl not in models:
            print(f"  [skip] {label}: missing per-chunk data")
            continue
        keys = sorted(set(models[treat]) & set(models[ctrl]))
        if not keys:
            print(f"  [skip] {label}: no overlapping units")
            continue
        diff = np.array([models[treat][k][KEY] - models[ctrl][k][KEY] for k in keys])
        m, lo, hi = ci(diff, rng, a.n_boot)
        win = float((diff > 0).mean())
        out["contrasts"][f"{treat}__vs__{ctrl}"] = {
            "label": label, "treatment": treat, "control": ctrl,
            "mean_diff": m, "lo": lo, "hi": hi, "n_pairs": len(keys),
            "frac_improved": win,
            "excludes_zero": bool(lo > 0 or hi < 0)}
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {star} {label:34s} {m:+.2f} dB  95% CI [{lo:+.2f}, {hi:+.2f}]"
              f"  {win*100:.0f}% of {len(keys)} units improved")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("* = 95% CI excludes zero. Evaluation variance only, not training-seed variance.")


if __name__ == "__main__":
    main()
