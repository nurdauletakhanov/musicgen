"""Effect of recentering the latent origin on stem-subtraction scores.

For each matched pair (treatment = mixing-trained model, control = its
no-mixing twin at the same compression rate) this computes, on the SAME
(track, stem, chunk) units:

  raw advantage        = sub(treat)          - sub(ctrl)
  corrected advantage  = sub(treat+origin)   - sub(ctrl+origin)
  change               = corrected advantage - raw advantage      (per unit)

plus per-stem corrected advantages and the per-model gap to each model's own
reconstruction reference ("ceil"). CIs resample whole tracks; they are a
conditional diagnostic on these checkpoints and this test audio, and do not
cover checkpoint selection or training-seed variance.
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

PAIRS = [
    ("v2.1-decmix", "v2.0-continued", "decode-mixing, 7.66x"),
    ("v2.2-decmix-disc", "v2.0-continued", "decode-mixing + disc, 7.66x"),
    ("v3.1-decmix-disc-d64", "v3.0-baseline-d64", "decode-mixing + disc, 15.3x"),
]
N_BOOT, SEED = 10000, 0


def cluster_ci(vals, tracks, rng, n_boot=N_BOOT):
    vals = np.asarray(vals, float); tracks = np.asarray(tracks)
    ids = np.unique(tracks)
    sums = np.array([vals[tracks == c].sum() for c in ids])
    cnts = np.array([(tracks == c).sum() for c in ids])
    draw = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    means = sums[draw].sum(axis=1) / cnts[draw].sum(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-chunk", default="evaluation/v2_metrics/per_chunk/*_per_chunk.json")
    ap.add_argument("--activity", default=None,
                    help="activity.json from evaluation.holdout_activity; when "
                         "given, the pre-registered silent-target rule is "
                         "applied (RMS >= -50 dBFS and target/mix >= -30 dB)")
    ap.add_argument("--min-rms-dbfs", type=float, default=-50.0)
    ap.add_argument("--min-target-to-mix-db", type=float, default=-30.0)
    ap.add_argument("--out", default="evaluation/v2_metrics/origin_effect.json")
    a = ap.parse_args()
    M = {}
    for p in sorted(glob.glob(a.per_chunk)):
        name = re.sub(r"_per_chunk\.json$", "", os.path.basename(p))
        M[name] = {(r["track"], r["stem"], r["chunk"]): r for r in json.load(open(p))["records"]}
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
    out = {"n_boot": N_BOOT, "pairs": {}, "models": {}}

    for name, recs in sorted(M.items()):
        keys = sorted(recs); tracks = [k[0] for k in keys]
        sub = np.array([recs[k]["sub"] for k in keys])
        gap = np.array([recs[k]["ceil"] - recs[k]["sub"] for k in keys])
        m, lo, hi, nt = cluster_ci(sub, tracks, rng)
        gm, glo, ghi, _ = cluster_ci(gap, tracks, rng)
        ref = np.array([recs[k]["ceil"] for k in keys])
        out["models"][name] = {"reference": float(ref.mean()), "sub": m, "sub_lo": lo, "sub_hi": hi,
                               "gap_to_reference": gm, "gap_lo": glo, "gap_hi": ghi,
                               "n_units": len(keys), "n_tracks": nt}
        print(f"{name:32s} sub {m:+.2f} [{lo:+.2f},{hi:+.2f}]  gap {gm:+.2f} [{glo:+.2f},{ghi:+.2f}]")

    print("\nchange in between-model advantage after recentering:")
    for treat, ctrl, label in PAIRS:
        need = [treat, ctrl, treat + "+origin", ctrl + "+origin"]
        if any(n not in M for n in need):
            print(f"  [skip] {label}"); continue
        keys = sorted(set.intersection(*[set(M[n]) for n in need]))
        tracks = np.array([k[0] for k in keys]); stems = np.array([k[1] for k in keys])
        raw = np.array([M[treat][k]["sub"] - M[ctrl][k]["sub"] for k in keys])
        cor = np.array([M[treat + "+origin"][k]["sub"] - M[ctrl + "+origin"][k]["sub"] for k in keys])
        chg = cor - raw
        rm, rlo, rhi, nt = cluster_ci(raw, tracks, rng)
        cm, clo, chi, _ = cluster_ci(cor, tracks, rng)
        dm, dlo, dhi, _ = cluster_ci(chg, tracks, rng)
        per_track = np.array([chg[tracks == t].mean() for t in np.unique(tracks)])
        neg = int((per_track < 0).sum())
        per_stem = {s: float(cor[stems == s].mean()) for s in sorted(set(stems))}
        tr_ids = list(np.unique(tracks))
        per_track_raw = [float(raw[tracks == t].mean()) for t in tr_ids]
        per_track_cor = [float(cor[tracks == t].mean()) for t in tr_ids]
        out["pairs"][f"{treat}__vs__{ctrl}"] = {
            "label": label, "treatment": treat, "control": ctrl,
            "raw_advantage": rm, "raw_lo": rlo, "raw_hi": rhi,
            "corrected_advantage": cm, "corrected_lo": clo, "corrected_hi": chi,
            "change": dm, "change_lo": dlo, "change_hi": dhi,
            "n_tracks": nt, "tracks_reduced": neg,
            "corrected_by_stem": per_stem,
            "per_track_raw": per_track_raw, "per_track_corrected": per_track_cor,
            "corrected_excludes_zero": bool(clo > 0 or chi < 0)}
        print(f"  {label:28s} raw {rm:+.2f} [{rlo:+.2f},{rhi:+.2f}] -> corrected {cm:+.2f} "
              f"[{clo:+.2f},{chi:+.2f}]  change {dm:+.2f} [{dlo:+.2f},{dhi:+.2f}]  "
              f"reduced on {neg}/{nt} tracks")
        print(f"      corrected by stem: " + ", ".join(f"{s} {v:+.2f}" for s, v in per_stem.items()))
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
