"""Run per-stem latent-subtraction eval across the v2/v3 + M2L Phase 2 models.

Shells out to ``evaluation.compute_subtraction`` for the v2/v3 checkpoints and
to ``evaluation.m2l_run_subtraction`` for M2L. Writes one JSON per model into
``evaluation/v2_metrics/``.

Usage:
  # full eval, all 4 models, default 30 chunks/track on the 50-track test split
  python -m scripts.run_subtraction

  # smoke (5 tracks per model, fast)
  python -m scripts.run_subtraction --max-tracks 5

  # subset
  python -m scripts.run_subtraction --only v3.1-decmix-disc-d64

Already-existing outputs are skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "evaluation" / "v2_metrics"

# Music2Latent Phase-2 checkpoint (sibling repo). Override with:
#   export MUSICGEN_M2L_CHECKPOINT=/path/to/model_..._iters_50000.pt
# Only the "m2l-phase2" row needs this; the v2/v3 rows run without it.
M2L_PHASE2_CKPT = os.environ.get(
    "MUSICGEN_M2L_CHECKPOINT",
    str(Path(os.environ.get("MUSICGEN_M2L_REPO", REPO.parent / "music2latent-mix"))
        / "checkpoints" / "mix_phase2_decmix_consmix"
        / "2026-05-06_14-11-27"
        / "model_fid_-1.0_loss_119.60_iters_50000.pt"))

# (name, kind, payload)
#   kind="v2"   payload=path to checkpoint dir under REPO/checkpoints/
#   kind="m2l"  payload=absolute path to .pt file
MODELS = [
    ("v2.0-continued",        "v2",  REPO / "checkpoints" / "v2.0-continued"),
    ("v2.1-decmix",           "v2",  REPO / "checkpoints" / "v2.1-decmix"),
    ("v2.2-decmix-disc",      "v2",  REPO / "checkpoints" / "v2.2-decmix-disc"),
    ("v3.0-baseline-d64",     "v2",  REPO / "checkpoints" / "v3.0-baseline-d64"),
    ("v3.1-decmix-disc-d64",  "v2",  REPO / "checkpoints" / "v3.1-decmix-disc-d64"),
    ("m2l-phase2",            "m2l",  M2L_PHASE2_CKPT),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only the given model names (default: all)")
    ap.add_argument("--checkpoint-name", default="best.pth",
                    help="Which checkpoint file inside each v2 ckpt dir to use")
    ap.add_argument("--chunks-per-track", type=int, default=30)
    ap.add_argument("--max-tracks", type=int, default=None,
                    help="Cap MUSDB tracks per model (smoke test)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--m2l-batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--musdb-dir", type=str,
                    default=str(REPO / "dataset" / "musdb18" / "test"))
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if output JSON already exists")
    args = ap.parse_args()

    names = [m[0] for m in MODELS]
    targets = args.only if args.only else names
    unknown = [n for n in targets if n not in names]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}. valid: {names}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, kind, payload in MODELS:
        if name not in targets:
            continue
        out = OUT_DIR / f"{name}_subtraction.json"
        if out.exists() and not args.force:
            print(f"[skip] {name}: {out} already exists (use --force to redo)")
            summary.append((name, "skipped-exists"))
            continue

        if kind == "v2":
            ckpt_dir = Path(payload)
            cfg = ckpt_dir / "config.yaml"
            ckpt = ckpt_dir / args.checkpoint_name
            if not cfg.exists():
                print(f"[skip] {name}: missing {cfg}")
                summary.append((name, "missing-config"))
                continue
            if not ckpt.exists():
                print(f"[skip] {name}: missing {ckpt}")
                summary.append((name, "missing-ckpt"))
                continue
            cmd = [
                sys.executable, "-m", "evaluation.compute_subtraction",
                "--config", str(cfg),
                "--checkpoint", str(ckpt),
                "--out", str(out),
                "--musdb-dir", args.musdb_dir,
                "--chunks-per-track", str(args.chunks_per_track),
                "--batch-size", str(args.batch_size),
                "--seed", str(args.seed),
            ]
        elif kind == "m2l":
            if not Path(payload).exists():
                print(f"[skip] {name}: missing M2L checkpoint {payload}")
                summary.append((name, "missing-m2l-ckpt"))
                continue
            cmd = [
                sys.executable, "-m", "evaluation.m2l_run_subtraction",
                "--m2l-checkpoint", payload,
                "--out", str(out),
                "--musdb-dir", args.musdb_dir,
                "--chunks-per-track", str(args.chunks_per_track),
                "--batch-size", str(args.m2l_batch_size),
                "--seed", str(args.seed),
            ]
        else:
            raise SystemExit(f"unknown kind: {kind}")

        if args.max_tracks is not None:
            cmd += ["--max-tracks", str(args.max_tracks)]

        print("\n" + "=" * 70)
        print(f"[run ] {name}  ({kind})")
        print("       " + " ".join(cmd))
        print("=" * 70)
        rc = subprocess.call(cmd, cwd=str(REPO))
        summary.append((name, "ok" if rc == 0 else f"failed-rc{rc}"))

    print("\n=== summary ===")
    for name, status in summary:
        print(f"  {name:30s} {status}")
    failed = [n for n, s in summary if s.startswith("failed")]
    if failed:
        sys.exit(f"\n{len(failed)} run(s) failed: {failed}")


if __name__ == "__main__":
    main()
