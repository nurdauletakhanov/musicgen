"""
Alpha sweep of the mixing-equivariance metrics for the paper's key models.

For every (model, alpha) pair, shells out to the matching eval module
(`evaluation.compute_mixing_metrics` for v2/v3 checkpoints,
`evaluation.m2l_run_mixing` for Music2Latent) and writes one JSON per pair
into evaluation/v2_metrics/alpha_sweep/. Afterwards aggregates the headline
metrics into evaluation/v2_metrics/alpha_sweep_summary.json.

The sweep is the paper's protocol figure: under the shared-noise protocol
sdr_lin should be roughly alpha-independent, while a phase-variance-dominated
protocol shows a U-shape with the minimum at alpha=0.5.

Usage:
  python -m scripts.run_alpha_sweep                  # all models, all alphas
  python -m scripts.run_alpha_sweep --only v2.0-continued v2.2-decmix-disc
  python -m scripts.run_alpha_sweep --max-batches 5  # smoke test

Already-existing outputs are skipped unless --force is passed.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
M2L_REPO = Path("d:/projects/music2latent-mix")
OUT_DIR = REPO / "evaluation" / "v2_metrics" / "alpha_sweep"
SUMMARY_PATH = REPO / "evaluation" / "v2_metrics" / "alpha_sweep_summary.json"

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]

# kind "v2": config/checkpoint under checkpoints/<name>/.
# kind "m2l": checkpoint resolved from ckpt_dir (newest *_ema.pt under the
#             run's timestamped subdirs); None = published M2L checkpoint.
MODELS = [
    {"kind": "v2", "name": "v2.0-continued"},
    {"kind": "v2", "name": "v2.2-decmix-disc"},
    {"kind": "v2", "name": "v3.0-baseline-d64"},
    {"kind": "v2", "name": "v3.1-decmix-disc-d64"},
    {"kind": "m2l", "name": "m2l-phase0", "ckpt_dir": None},
    {"kind": "m2l", "name": "m2l-phase05",
     "ckpt_dir": M2L_REPO / "checkpoints" / "mix_phase05_control"},
    {"kind": "m2l", "name": "m2l-phase2",
     "ckpt_dir": M2L_REPO / "checkpoints" / "mix_phase2_decmix_consmix"},
]

SUMMARY_METRICS = ("sdr_lin", "sdr_lin_gt", "sdr_rec", "mix_rate", "l_lat")


def _resolve_m2l_ckpt(ckpt_dir):
    """Newest EMA-merged checkpoint under <ckpt_dir>/<timestamp>/."""
    if ckpt_dir is None:
        return None  # published checkpoint, EncoderDecoder default
    hits = sorted(glob.glob(str(ckpt_dir / "*" / "*_ema.pt")),
                  key=os.path.getmtime)
    return hits[-1] if hits else "MISSING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only the given model names")
    ap.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    names = [m["name"] for m in MODELS]
    targets = args.only if args.only else names
    unknown = [n for n in targets if n not in names]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}. valid: {names}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_runs = []
    for model in MODELS:
        if model["name"] not in targets:
            continue
        name = model["name"]
        for alpha in args.alphas:
            out = OUT_DIR / f"{name}_a{alpha:g}_mixing.json"
            if out.exists() and not args.force:
                print(f"[skip] {name} a={alpha:g}: exists")
                summary_runs.append((name, alpha, "skipped-exists"))
                continue

            if model["kind"] == "v2":
                ckpt_dir = REPO / "checkpoints" / name
                cfg = ckpt_dir / "config.yaml"
                ckpt = ckpt_dir / "best.pth"
                if not cfg.exists() or not ckpt.exists():
                    print(f"[skip] {name}: missing config/checkpoint "
                          f"(not trained yet?)")
                    summary_runs.append((name, alpha, "missing"))
                    break  # no point trying other alphas
                cmd = [
                    sys.executable, "-m", "evaluation.compute_mixing_metrics",
                    "--config", str(cfg), "--checkpoint", str(ckpt),
                    "--out", str(out), "--alpha", str(alpha),
                    "--batch-size", str(args.batch_size),
                    "--seed", str(args.seed),
                ]
            else:
                ckpt = _resolve_m2l_ckpt(model["ckpt_dir"])
                if ckpt == "MISSING":
                    print(f"[skip] {name}: no *_ema.pt under "
                          f"{model['ckpt_dir']} (not trained/converted yet?)")
                    summary_runs.append((name, alpha, "missing"))
                    break
                cmd = [
                    sys.executable, "-m", "evaluation.m2l_run_mixing",
                    "--out", str(out), "--alpha", str(alpha),
                    "--batch-size", str(min(args.batch_size, 16)),
                    "--seed", str(args.seed),
                ]
                if ckpt is not None:
                    cmd += ["--m2l-checkpoint", str(ckpt)]
            if args.max_batches is not None:
                cmd += ["--max-batches", str(args.max_batches)]

            print("\n" + "=" * 70)
            print(f"[run ] {name}  alpha={alpha:g}")
            print("       " + " ".join(cmd))
            print("=" * 70)
            rc = subprocess.call(cmd, cwd=str(REPO))
            summary_runs.append((name, alpha, "ok" if rc == 0 else f"failed-rc{rc}"))

    # ---- aggregate everything present into one summary JSON
    agg = {}
    for f in sorted(OUT_DIR.glob("*_mixing.json")):
        stem = f.stem[: -len("_mixing")]
        name, _, a = stem.rpartition("_a")
        with open(f) as fh:
            data = json.load(fh)
        metrics = data.get("metrics", {})
        agg.setdefault(name, {})[a] = {
            k: metrics.get(k, {}).get("all") for k in SUMMARY_METRICS
        }
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(agg, fh, indent=2, sort_keys=True)
    print(f"\nwrote {SUMMARY_PATH} ({len(agg)} models)")

    print("\n=== summary ===")
    for name, alpha, status in summary_runs:
        print(f"  {name:24s} a={alpha:<4g} {status}")
    failed = [(n, a) for n, a, s in summary_runs if s.startswith("failed")]
    if failed:
        sys.exit(f"\n{len(failed)} run(s) failed: {failed}")


if __name__ == "__main__":
    main()
