#!/bin/bash
#SBATCH --job-name=gain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
# Gain-preservation check: SI-SDR is scale-invariant, so report the fitted
# scale (dB) of g(z̄) vs x̄ on a source-balanced subsample for the matched pairs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A CFG=(
  [v2.0-continued]=configs/experiments/v2/v2.0_continued.yaml
  [v2.1-decmix]=configs/experiments/v2/v2.1_decmix.yaml
  [v3.0-baseline-d64]=configs/experiments/v3/v3.0_baseline_d64.yaml
  [v3.1-decmix-disc-d64]=configs/experiments/v3/v3.1_decmix_disc_d64.yaml )
mkdir -p evaluation/v2_metrics/gain
for r in v2.0-continued v2.1-decmix v3.0-baseline-d64 v3.1-decmix-disc-d64; do
  out=evaluation/v2_metrics/gain/${r}_mixing_gain.json
  [ -f "$out" ] && { echo "[skip] $r"; continue; }
  cfg="checkpoints/$r/config.yaml"; [ -f "$cfg" ] || cfg="${CFG[$r]}"
  echo "[$(date +%T)] $r"
  .venv/bin/python -m evaluation.compute_mixing_metrics --config "$cfg" \
    --checkpoint "checkpoints/$r/best.pth" --out "$out" --per-source 1000 --seed 0
done
.venv/bin/python - <<'PY'
import json
for r in ["v2.0-continued","v2.1-decmix","v3.0-baseline-d64","v3.1-decmix-disc-d64"]:
    m=json.load(open(f"evaluation/v2_metrics/gain/{r}_mixing_gain.json"))["metrics"]
    print(f"{r:24s} sdr_lin_gt {m['sdr_lin_gt']['all']:+.2f}  gain(g(zbar) vs xbar) {m['gain_lin_gt']['all']:+.2f} dB  |gain| {m['abs_gain_lin_gt']['all']:.2f} dB   recon gain {m['gain_rec']['all']:+.2f} |.| {m['abs_gain_rec']['all']:.2f}")
PY
