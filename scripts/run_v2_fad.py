"""
Run per-source FAD (CLAP-based) on every v2.0–v2.6 model.

Iterates the seven v2.x checkpoint dirs, shells out to
`python -m evaluation.compute_fad` for each, and writes one JSON per model
into evaluation/v2_metrics/.

Usage:
  # full eval, all 7 models
  python -m scripts.run_v2_fad

  # smoke test (3 tracks per source per model)
  python -m scripts.run_v2_fad --max-tracks-per-source 3

  # subset of models
  python -m scripts.run_v2_fad --only v2.1-decmix v2.2-decmix-disc

Already-existing outputs are skipped unless --force is passed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODELS = [
    "v2.0-continued",
    "v2.1-decmix",
    "v2.2-decmix-disc",
    "v2.3-encmix-g5",
    "v2.4-encmix-g10",
    "v2.5-encmix-g20",
    "v2.6-decmix-frozenenc",
    "v3.0-baseline-d64",
    "v3.1-decmix-disc-d64",
]

OUT_DIR = REPO / "evaluation" / "v2_metrics"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only the given model names (default: all 7)")
    ap.add_argument("--checkpoint-name", default="best.pth",
                    help="Which checkpoint file inside each ckpt dir to use")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Decode batch size inside compute_fad")
    ap.add_argument("--max-tracks-per-source", type=int, default=None,
                    help="Cap tracks per source per model (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if output JSON already exists")
    args = ap.parse_args()

    targets = args.only if args.only else MODELS
    unknown = [m for m in targets if m not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}. valid: {MODELS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for name in targets:
        ckpt_dir = REPO / "checkpoints" / name
        cfg = ckpt_dir / "config.yaml"
        ckpt = ckpt_dir / args.checkpoint_name
        out = OUT_DIR / f"{name}_fad.json"

        if not cfg.exists():
            print(f"[skip] {name}: missing {cfg}")
            summary.append((name, "missing-config"))
            continue
        if not ckpt.exists():
            print(f"[skip] {name}: missing {ckpt}")
            summary.append((name, "missing-ckpt"))
            continue
        if out.exists() and not args.force:
            print(f"[skip] {name}: {out} already exists (use --force to redo)")
            summary.append((name, "skipped-exists"))
            continue

        cmd = [
            sys.executable, "-m", "evaluation.compute_fad",
            "--config", str(cfg),
            "--checkpoint", str(ckpt),
            "--out", str(out),
            "--batch-size", str(args.batch_size),
        ]
        if args.max_tracks_per_source is not None:
            cmd += ["--max-tracks-per-source", str(args.max_tracks_per_source)]

        print("\n" + "=" * 70)
        print(f"[run ] {name}")
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
