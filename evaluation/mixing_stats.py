"""Paired intervals for the convex-mixing contrasts.

Every mixing unit is a *pair* of recordings: chunk i from one track is mixed
with chunk j from another. A one-way cluster bootstrap over "the" recording is
therefore not valid, because it leaves the dependence through the second
member unaccounted for. We resample recordings with replacement and keep a
pair only when BOTH of its recordings were drawn (a dyadic bootstrap), and
report the unit-level interval alongside, labelled as a lower bound.

Pairing is a fixed function of the seed and the batch index, so all models see
identical pairs and every contrast is paired unit by unit.

Usage:
  python -m evaluation.mixing_stats \
      --per-unit 'evaluation/v2_metrics/per_chunk/*_mixing_per_unit.json' \
      --out evaluation/v2_metrics/mixing_stats.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from evaluation.paired_stats import ci, cluster_ci, dyadic_cluster_ci

CONTRASTS = [
    ("v2.1-decmix", "v2.0-continued", "L_dec vs no mix (7.66x)"),
    ("v2.2-decmix-disc", "v2.0-continued", "L_dec+disc vs no mix (7.66x)"),
    ("v2.2-decmix-disc", "v2.1-decmix", "disc-on-mix vs L_dec alone"),
    ("v3.1-decmix-disc-d64", "v3.0-baseline-d64", "L_dec+disc vs no mix (15.3x)"),
]
METRICS = ["sdr_lin_gt", "sdr_rec", "mix_rate", "l_lat", "sdr_lin",
           "l_lat_abs", "l_lat_span", "l_lat_centered"]
N_BOOT, SEED = 10000, 0


def load(paths):
    out = {}
    for p in paths:
        name = re.sub(r"_mixing_per_unit\.json$", "", os.path.basename(p))
        recs = json.load(open(p))["records"]
        out[name] = {(r["track"], r["chunk"]): r for r in recs}
        print(f"  {name}: {len(recs)} units")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-unit", nargs="+",
                    default=["evaluation/v2_metrics/per_chunk/*_mixing_per_unit.json"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    paths = [p for pat in a.per_unit for p in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no per-unit files matched: {a.per_unit}")
    print("loading:")
    M = load(paths)
    rng = np.random.default_rng(SEED)
    out = {"n_boot": a.n_boot, "models": {}, "contrasts": {}}

    for name, recs in sorted(M.items()):
        keys = sorted(recs)
        left = [k[0] for k in keys]
        right = [recs[k]["pair_track"] for k in keys]
        out["models"][name] = {}
        for met in METRICS:
            if met not in recs[keys[0]]:
                continue
            vals = np.array([recs[k][met] for k in keys])
            m, lo, hi, nrec = dyadic_cluster_ci(vals, left, right, rng, a.n_boot)
            out["models"][name][met] = {"mean": m, "lo": lo, "hi": hi,
                                        "n_units": len(keys), "n_recordings": nrec}
        g = out["models"][name].get("sdr_lin_gt", {})
        print(f"{name:24s} sdr_lin_gt {g.get('mean', float('nan')):+.2f} "
              f"[{g.get('lo', float('nan')):+.2f}, {g.get('hi', float('nan')):+.2f}] "
              f"over {g.get('n_recordings', 0)} recordings")

    print("\npaired contrasts (identical pairs; dyadic recording bootstrap):")
    for treat, ctrl, label in CONTRASTS:
        if treat not in M or ctrl not in M:
            print(f"  [skip] {label}: missing per-unit data")
            continue
        keys = sorted(set(M[treat]) & set(M[ctrl]))
        if not keys:
            print(f"  [skip] {label}: no overlapping units")
            continue
        # Both models must have been paired the same way for this to be paired.
        mismatch = sum(1 for k in keys
                       if M[treat][k]["pair_track"] != M[ctrl][k]["pair_track"])
        left = [k[0] for k in keys]
        right = [M[treat][k]["pair_track"] for k in keys]
        entry = {"label": label, "treatment": treat, "control": ctrl,
                 "n_units": len(keys), "pairing_mismatches": mismatch}
        for met in METRICS:
            if met not in M[treat][keys[0]]:
                continue
            d = np.array([M[treat][k][met] - M[ctrl][k][met] for k in keys])
            m, lo, hi, nrec = dyadic_cluster_ci(d, left, right, rng, a.n_boot)
            _, ulo, uhi = ci(d, rng, a.n_boot)
            entry[met] = {"mean_diff": m, "lo": lo, "hi": hi,
                          "excludes_zero": bool(lo > 0 or hi < 0),
                          "unit_lo": ulo, "unit_hi": uhi, "n_recordings": nrec}
        out["contrasts"][f"{treat}__vs__{ctrl}"] = entry
        g = entry.get("sdr_lin_gt", {})
        star = "*" if g.get("excludes_zero") else " "
        print(f"  {star} {label:32s} sdr_lin_gt {g.get('mean_diff', float('nan')):+.2f} dB  "
              f"dyadic CI [{g.get('lo', float('nan')):+.2f}, {g.get('hi', float('nan')):+.2f}]  "
              f"(unit CI [{g.get('unit_lo', float('nan')):+.2f}, {g.get('unit_hi', float('nan')):+.2f}])")
        if mismatch:
            print(f"      WARNING: {mismatch} units paired differently between the two runs; "
                  f"the contrast is not fully paired")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("* = 95% dyadic CI excludes zero. Intervals cover evaluation variance "
          "over recordings, not training-seed variance or checkpoint selection.")


if __name__ == "__main__":
    main()
