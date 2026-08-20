"""
Phase-3 M2L re-evaluation batch: consistent EMA weights + new metrics.

Why this exists: the original M2L phase evals (a) used RAW fine-tune weights
while Phase 0 used the published EMA-merged weights, and (b) predate the
sdr_lin_gt / sdr_dd metrics. This driver re-runs everything against the
*_ema.pt checkpoints so the whole M2L table is apples-to-apples.

Outputs use an `_ema` suffix (e.g. m2l_phase2_ema_mixing.json) so the old
JSONs remain for comparison; RESULTS.md and the paper tables should use the
_ema files. Phase 0 has no _ema checkpoint (the published file already
stores EMA weights) but is re-run for the sdr_lin_gt column.

Usage:
  python -m scripts.run_m2l_phase3                 # everything missing
  python -m scripts.run_m2l_phase3 --only phase2   # one phase
  python -m scripts.run_m2l_phase3 --max-batches 2 # smoke test (mixing only)

Skips outputs that already exist unless --force.
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Music2Latent fine-tuning lives in a sibling repo. Override with:
#   export MUSICGEN_M2L_REPO=/path/to/music2latent-mix
M2L_REPO = Path(os.environ.get("MUSICGEN_M2L_REPO", REPO.parent / "music2latent-mix"))
OUT_DIR = REPO / "evaluation" / "v2_metrics"

# phase -> checkpoint dir under music2latent-mix/checkpoints (None = published)
PHASES = {
    "phase0": None,
    "phase05": "mix_phase05_control",
    "phase1": "mix_phase1_encmix",
    "phase2": "mix_phase2_decmix_consmix",
    "phase2b": "mix_phase2b_decmix_w5",
}

# which evals to run per phase: mixing for all (sdr_lin_gt), FAD only where
# the existing JSON was computed with raw weights, subtraction for the
# downstream table rows (phase0 baseline + phase2 headline; phase05 done).
RUN_FAD = {"phase1", "phase2", "phase2b"}
RUN_SUBTRACTION = {"phase0", "phase2"}


def _resolve_ckpt(phase: str):
    sub = PHASES[phase]
    if sub is None:
        return None
    hits = sorted(glob.glob(str(M2L_REPO / "checkpoints" / sub / "*" / "*_ema.pt")),
                  key=os.path.getmtime)
    return hits[-1] if hits else "MISSING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, choices=list(PHASES))
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Smoke test: cap mixing-eval batches and skip FAD/subtraction")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    targets = args.only if args.only else list(PHASES)
    jobs = []  # (label, out_path, cmd)

    for phase in targets:
        ckpt = _resolve_ckpt(phase)
        if ckpt == "MISSING":
            print(f"[skip] {phase}: no *_ema.pt found (run make_ema_checkpoint.py first)")
            continue
        ckpt_args = ["--m2l-checkpoint", str(ckpt)] if ckpt else []

        out = OUT_DIR / f"m2l_{phase}_ema_mixing.json"
        cmd = [sys.executable, "-m", "evaluation.m2l_run_mixing",
               "--out", str(out), *ckpt_args]
        if args.max_batches is not None:
            cmd += ["--max-batches", str(args.max_batches)]
        jobs.append((f"{phase} mixing", out, cmd))

        if args.max_batches is None and phase in RUN_FAD:
            out = OUT_DIR / f"m2l_{phase}_ema_fad.json"
            jobs.append((f"{phase} fad", out,
                         [sys.executable, "-m", "evaluation.m2l_run_fad",
                          "--out", str(out), *ckpt_args]))

        if args.max_batches is None and phase in RUN_SUBTRACTION:
            out = OUT_DIR / f"m2l_{phase}_ema_subtraction.json"
            jobs.append((f"{phase} subtraction", out,
                         [sys.executable, "-m", "evaluation.m2l_run_subtraction",
                          "--out", str(out), *ckpt_args]))

    summary = []
    for label, out, cmd in jobs:
        if out.exists() and not args.force:
            print(f"[skip] {label}: {out.name} exists")
            summary.append((label, "skipped-exists"))
            continue
        print("\n" + "=" * 70)
        print(f"[run ] {label}")
        print("       " + " ".join(cmd))
        print("=" * 70)
        rc = subprocess.call(cmd, cwd=str(REPO))
        summary.append((label, "ok" if rc == 0 else f"failed-rc{rc}"))

    print("\n=== summary ===")
    for label, status in summary:
        print(f"  {label:24s} {status}")
    failed = [l for l, s in summary if s.startswith("failed")]
    if failed:
        sys.exit(f"\n{len(failed)} job(s) failed: {failed}")


if __name__ == "__main__":
    main()
