"""Paired bootstrap confidence intervals for the stem-subtraction results.

The subtraction eval measures every model on the *same* (track, chunk, stem)
units: chunk selection is a pure function of the seed and the sorted track
list, so measurements align one-to-one across models. That makes a paired
analysis valid and much tighter than comparing two independent means, because
"some chunks are simply harder" cancels out.

For each model we report a percentile bootstrap CI on the mean SI-SDR, and for
each contrast (a model against its matched control) a CI on the *paired mean
difference*, plus the fraction of units that improved.

Resampling unit. Units from the same recording are correlated (adjacent
windows, same mix), so the CI of record resamples WHOLE TRACKS (a paired
cluster bootstrap: draw the 49 test tracks with replacement, keep every unit
of each drawn track, recompute the mean paired difference). This is the right
unit when the inferential target is new tracks. Unit-level resampling is also
reported, labelled as such; it is narrower and should be read as a lower bound
on the uncertainty.

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
] + [
    # Origin-corrected subtraction g(f(mix) - f(stem) + f(0)) vs raw, per model:
    # does a latent offset contribute to the subtraction error?
    (f"{r}+origin", r, f"origin-corrected vs raw: {r}")
    for r in ("v2.0-continued", "v2.1-decmix", "v2.2-decmix-disc",
              "v3.0-baseline-d64", "v3.1-decmix-disc-d64",
              "v1.1", "m2l_phase0_ema", "m2l_phase2_ema")
]
KEY = "sub"          # metric of record: latent subtraction vs ground truth

# The paper's corrected comparisons: matched contrasts on origin-corrected
# subtraction, and corrected latent subtraction against the no-arithmetic
# comparators recorded in the raw files (same units).
CORRECTED_CONTRASTS = [
    ("v2.1-decmix+origin", "v2.0-continued+origin", "corrected: L_dec vs no mix (7.66x)"),
    ("v2.2-decmix-disc+origin", "v2.0-continued+origin", "corrected: L_dec+disc vs no mix (7.66x)"),
    ("v3.1-decmix-disc-d64+origin", "v3.0-baseline-d64+origin", "corrected: mix vs no mix (15.3x)"),
]

# Comparators recorded alongside "sub" in the same per-chunk files (when the
# eval was run with the comparator code): latent subtraction vs ...
COMPARATOR_CONTRASTS = [
    ("wavsub", "latent subtraction vs decode-then-subtract g(f(mix))-g(f(stem))"),
    ("aemix", "latent subtraction vs autoencoded unchanged mixture g(f(mix))"),
    ("identity", "latent subtraction vs unchanged raw mixture"),
]
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
    """Percentile bootstrap CI for the mean, resampling individual units."""
    vals = np.asarray(vals, dtype=np.float64)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cluster_ci(vals, clusters, rng, n_boot=N_BOOT):
    """Percentile bootstrap CI for the mean, resampling whole clusters (tracks).

    Draw clusters with replacement; each draw contributes all of its units, so
    the resampled mean is the unit-weighted mean over the drawn clusters.
    """
    vals = np.asarray(vals, dtype=np.float64)
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    sums = np.array([vals[clusters == c].sum() for c in ids])
    cnts = np.array([(clusters == c).sum() for c in ids])
    draw = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    means = sums[draw].sum(axis=1) / cnts[draw].sum(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int(len(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-chunk", nargs="+", required=True)
    ap.add_argument("--activity", default=None,
                    help="activity.json from evaluation.holdout_activity; when "
                         "given, the pre-registered silent-target rule is "
                         "applied (RMS >= -50 dBFS and target/mix >= -30 dB)")
    ap.add_argument("--min-rms-dbfs", type=float, default=-50.0)
    ap.add_argument("--min-target-to-mix-db", type=float, default=-30.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    paths = [p for pat in a.per_chunk for p in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no per-chunk files matched: {a.per_chunk}")
    print("loading:")
    models = load(paths)
    if a.activity:
        act = json.load(open(a.activity))
        keep = {tuple(k.split("|")[:2]) + (int(k.split("|")[2]),) for k, v in act.items()
                if v["rms_dbfs"] >= a.min_rms_dbfs
                and v["target_to_mix_db"] >= a.min_target_to_mix_db}
        before = sum(len(v) for v in M.values()) if "M" in dir() else sum(len(v) for v in models.values())
        tgt = M if "M" in dir() else models
        for name in list(tgt):
            tgt[name] = {k: r for k, r in tgt[name].items() if k in keep}
        after = sum(len(v) for v in tgt.values())
        print(f"silent-target rule: kept {after}/{before} unit-measurements "
              f"({100*after/max(before,1):.1f}%)")

    rng = np.random.default_rng(SEED)

    out = {"metric": KEY, "n_boot": a.n_boot, "models": {}, "contrasts": {}}

    for name, recs in sorted(models.items()):
        keys = sorted(recs)
        vals = [recs[k][KEY] for k in keys]
        tracks = [k[0] for k in keys]
        m, lo, hi = ci(vals, rng, a.n_boot)
        _, clo, chi, nt = cluster_ci(vals, tracks, rng, a.n_boot)
        out["models"][name] = {"mean": m, "lo": lo, "hi": hi, "n": len(vals),
                               "track_lo": clo, "track_hi": chi, "n_tracks": nt}
        print(f"{name:26s} {m:+.2f} dB  track-CI [{clo:+.2f}, {chi:+.2f}] "
              f"(unit-CI [{lo:+.2f}, {hi:+.2f}])  n={len(vals)} units / {nt} tracks")

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
        tracks = [k[0] for k in keys]
        m, lo, hi = ci(diff, rng, a.n_boot)
        _, clo, chi, nt = cluster_ci(diff, tracks, rng, a.n_boot)
        win = float((diff > 0).mean())
        # fraction of TRACKS whose mean difference is positive
        tw = float(np.mean([diff[np.array(tracks) == t].mean() > 0 for t in np.unique(tracks)]))
        out["contrasts"][f"{treat}__vs__{ctrl}"] = {
            "label": label, "treatment": treat, "control": ctrl,
            "mean_diff": m, "n_pairs": len(keys), "n_tracks": nt,
            "track_lo": clo, "track_hi": chi, "track_excludes_zero": bool(clo > 0 or chi < 0),
            "unit_lo": lo, "unit_hi": hi, "unit_excludes_zero": bool(lo > 0 or hi < 0),
            "frac_units_improved": win, "frac_tracks_improved": tw}
        star = "*" if (clo > 0 or chi < 0) else " "
        print(f"  {star} {label:34s} {m:+.2f} dB  track-CI [{clo:+.2f}, {chi:+.2f}] "
              f"(unit-CI [{lo:+.2f}, {hi:+.2f}])  {win*100:.0f}% units, {tw*100:.0f}% of {nt} tracks improved")

    print("\ncorrected matched contrasts (origin-corrected subtraction):")
    for treat, ctrl, label in CORRECTED_CONTRASTS:
        if treat not in models or ctrl not in models:
            print(f"  [skip] {label}: missing per-chunk data"); continue
        keys = sorted(set(models[treat]) & set(models[ctrl]))
        diff = np.array([models[treat][k][KEY] - models[ctrl][k][KEY] for k in keys])
        tracks = [k[0] for k in keys]
        m, lo, hi = ci(diff, rng, a.n_boot); _, clo, chi, nt = cluster_ci(diff, tracks, rng, a.n_boot)
        out["contrasts"][f"{treat}__vs__{ctrl}"] = {"label": label, "treatment": treat, "control": ctrl,
            "mean_diff": m, "n_pairs": len(keys), "n_tracks": nt, "track_lo": clo, "track_hi": chi,
            "track_excludes_zero": bool(clo > 0 or chi < 0), "unit_lo": lo, "unit_hi": hi}
        print(f"  {'*' if (clo>0 or chi<0) else ' '} {label:40s} {m:+.2f} dB  track-CI [{clo:+.2f}, {chi:+.2f}]")

    print("\ncorrected latent subtraction vs comparators (same units):")
    out["corrected_comparators"] = {}
    for name, recs in sorted(models.items()):
        if not name.endswith("+origin") or name[:-7] not in models: continue
        raw = models[name[:-7]]; keys = sorted(set(recs) & set(raw))
        if not keys or "wavsub" not in raw[keys[0]]: continue
        tracks = [k[0] for k in keys]
        for field, label in COMPARATOR_CONTRASTS:
            diff = np.array([recs[k][KEY] - raw[k][field] for k in keys])
            m, lo, hi = ci(diff, rng, a.n_boot); _, clo, chi, nt = cluster_ci(diff, tracks, rng, a.n_boot)
            out["corrected_comparators"][f"{name}__{field}"] = {"label": label, "model": name, "comparator": field,
                "mean_diff": m, "track_lo": clo, "track_hi": chi, "n_pairs": len(keys), "n_tracks": nt}
            print(f"  {'*' if (clo>0 or chi<0) else ' '} {name:30s} vs {field:8s}: {m:+.2f} dB  track-CI [{clo:+.2f}, {chi:+.2f}]")

    print("\ncomparators (same units; positive = latent subtraction better):")
    out["comparators"] = {}
    for name, recs in sorted(models.items()):
        keys = sorted(recs)
        if not keys or "wavsub" not in recs[keys[0]]:
            continue
        tracks = [k[0] for k in keys]
        for field, label in COMPARATOR_CONTRASTS:
            diff = np.array([recs[k][KEY] - recs[k][field] for k in keys])
            m, lo, hi = ci(diff, rng, a.n_boot)
            _, clo, chi, nt = cluster_ci(diff, tracks, rng, a.n_boot)
            base = float(np.mean([recs[k][field] for k in keys]))
            out["comparators"][f"{name}__{field}"] = {
                "label": label, "model": name, "comparator": field,
                "comparator_mean": base, "mean_diff": m,
                "track_lo": clo, "track_hi": chi, "unit_lo": lo, "unit_hi": hi,
                "n_pairs": len(keys), "n_tracks": nt}
            star = "*" if (clo > 0 or chi < 0) else " "
            print(f"  {star} {name:22s} vs {field:8s} (={base:+.2f} dB): {m:+.2f} dB  track-CI [{clo:+.2f}, {chi:+.2f}]")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("* = 95% track-cluster CI excludes zero. CIs cover evaluation variance "
          "(new tracks), not training-seed variance.")


if __name__ == "__main__":
    main()


def dyadic_cluster_ci(vals, left, right, rng, n_boot=N_BOOT):
    """Bootstrap CI for the mean of dyadic units (each unit involves TWO
    recordings, e.g. a mixing pair).

    A one-way cluster bootstrap is invalid here: resampling on one member of
    the pair leaves the dependence through the other member unaccounted for.
    We resample recordings with replacement and keep a pair only when BOTH of
    its recordings were drawn, which is the dyadic ("pigeonhole") bootstrap.

    vals  : per-unit values
    left  : recording id of the first member of each pair
    right : recording id of the second member
    """
    vals = np.asarray(vals, dtype=np.float64)
    left = np.asarray(left)
    right = np.asarray(right)
    ids = np.unique(np.concatenate([left, right]))
    pos = {c: i for i, c in enumerate(ids)}
    li = np.array([pos[c] for c in left])
    ri = np.array([pos[c] for c in right])
    means = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, len(ids), size=len(ids))
        # multiplicity of each recording in this resample
        mult = np.bincount(draw, minlength=len(ids))
        w = mult[li] * mult[ri]          # a pair enters once per (copy, copy)
        tot = w.sum()
        means[b] = float((vals * w).sum() / tot) if tot else np.nan
    means = means[~np.isnan(means)]
    return (float(vals.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), int(len(ids)))
