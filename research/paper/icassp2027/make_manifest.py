"""Freeze the exact inputs behind the submitted tables.

F13 of the repository audit: the result directory holds several generations of
experiment (EMA and non-EMA, the asymmetric-discriminator run, raw and
origin-corrected subtraction, filtered and unfiltered analyses). Filenames
alone do not say which ones the paper used, and the table generator discovers
inputs by glob, so an ambient file can change a rebuild.

This records, for every file the current tables read: its SHA-256, its size,
and the checkpoint the evaluator loaded. `make_tables.py --check-manifest`
then refuses to build if a canonical input is missing or has changed.

    python research/paper/icassp2027/make_manifest.py            # write
    python research/paper/icassp2027/make_manifest.py --verify   # check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
M = REPO / "evaluation" / "v2_metrics"
HOLDOUT = REPO / "evaluation" / "holdout_moisesdb"
OUT = HERE / "SUBMISSION_MANIFEST.json"

GAN_RUNS = ["v2.0-continued", "v2.1-decmix", "v2.2-decmix-disc",
            "v3.0-baseline-d64", "v3.1-decmix-disc-d64"]
ABLATION_RUNS = GAN_RUNS + ["v2.3-encmix-g5", "v2.4-encmix-g10",
                            "v2.5-encmix-g20", "v2.6-decmix-frozenenc"]
M2L_RUNS = ["m2l_phase0_ema", "m2l_phase05_ema", "m2l_phase2_ema"]
# FAD for the M2L rows lives under different basenames than the mixing JSONs
# (see the fad0/fad05 flags in make_tables.py), so list them explicitly.
M2L_FAD = ["m2l_phase0_fad.json", "m2l_phase05_control_fad.json",
           "m2l_phase2_ema_fad.json"]


def canonical_inputs() -> dict:
    """Table -> the files it reads. Anything not listed here is historical."""
    t = {}
    t["Table I (mechanism ablation)"] = (
        [f"evaluation/v2_metrics/{r}_mixing.json" for r in ABLATION_RUNS]
        + [f"evaluation/v2_metrics/{r}_fad.json" for r in ABLATION_RUNS])
    t["Table II (compression + architecture)"] = (
        [f"evaluation/v2_metrics/{r}_mixing.json" for r in GAN_RUNS + M2L_RUNS]
        + [f"evaluation/v2_metrics/{r}_fad.json" for r in GAN_RUNS]
        + [f"evaluation/v2_metrics/{f}" for f in M2L_FAD])
    t["Table III (stem subtraction)"] = (
        [f"evaluation/v2_metrics/per_chunk/{r}_per_chunk.json" for r in GAN_RUNS]
        + [f"evaluation/v2_metrics/per_chunk/{r}+origin_per_chunk.json" for r in GAN_RUNS])
    t["Table (origin intervention)"] = [
        "evaluation/v2_metrics/origin_effect.json",
        "evaluation/holdout_moisesdb/origin_effect.json",
    ]
    t["Held-out corpus and protocol"] = [
        "evaluation/holdout_moisesdb/activity.json",
        "evaluation/holdout_moisesdb/corpus_manifest.json",
        "evaluation/holdout_moisesdb/mixing_stats.json",
        "research/paper/icassp2027/HOLDOUT_PROTOCOL.md",
    ]
    return t


def digest(p: Path) -> dict | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    d = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
    if p.suffix == ".json":
        try:
            j = json.load(open(p))
            for k in ("checkpoint", "step", "alpha", "seed", "n_samples_seen",
                      "chunks_per_track", "n_boot"):
                if isinstance(j, dict) and k in j and not isinstance(j[k], (dict, list)):
                    d[k] = j[k]
        except Exception:
            pass
    return d


def build() -> dict:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                      text=True).strip()
    except Exception:
        rev = None
    tables, missing = {}, []
    for name, files in canonical_inputs().items():
        entry = {}
        for rel in files:
            d = digest(REPO / rel)
            if d is None:
                missing.append(rel)
            entry[rel] = d
        tables[name] = entry
    return {"evaluator_revision": rev, "missing": missing, "tables": tables}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="compare the tree against the manifest and exit "
                         "non-zero on any change")
    a = ap.parse_args()
    now = build()
    if not a.verify:
        OUT.write_text(json.dumps(now, indent=2))
        n = sum(len(v) for v in now["tables"].values())
        print(f"wrote {OUT.relative_to(REPO)}: {n} canonical inputs, "
              f"{len(now['missing'])} missing")
        for m in now["missing"]:
            print(f"  MISSING {m}")
        return
    if not OUT.exists():
        sys.exit(f"no manifest at {OUT}; run without --verify first")
    was = json.loads(OUT.read_text())
    problems = []
    for table, files in was["tables"].items():
        for rel, rec in files.items():
            cur = digest(REPO / rel)
            if rec is None:
                continue
            if cur is None:
                problems.append(f"missing: {rel}")
            elif cur["sha256"] != rec["sha256"]:
                problems.append(f"changed: {rel}")
    if problems:
        print("\n".join(problems))
        sys.exit(f"\n{len(problems)} canonical input(s) differ from the manifest")
    print(f"all canonical inputs match {OUT.name}")


if __name__ == "__main__":
    main()
