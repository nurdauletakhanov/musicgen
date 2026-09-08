#!/bin/bash
#SBATCH --job-name=origin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
# Origin-corrected latent subtraction, g(f(mix) - f(stem) + f(0)), for every GAN
# run, with per-chunk dumps; then the paired analysis including the
# corrected-vs-raw contrasts. Needs only MUSDB18 test + the checkpoints.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A CFG=(
  [v2.0-continued]=configs/experiments/v2/v2.0_continued.yaml
  [v2.1-decmix]=configs/experiments/v2/v2.1_decmix.yaml
  [v2.2-decmix-disc]=configs/experiments/v2/v2.2_decmix_disc.yaml
  [v3.0-baseline-d64]=configs/experiments/v3/v3.0_baseline_d64.yaml
  [v3.1-decmix-disc-d64]=configs/experiments/v3/v3.1_decmix_disc_d64.yaml )
for r in "${!CFG[@]}"; do
  out="evaluation/v2_metrics/per_chunk/${r}+origin_per_chunk.json"
  [ -f "$out" ] && { echo "[skip] $r"; continue; }
  cfg="checkpoints/$r/config.yaml"; [ -f "$cfg" ] || cfg="${CFG[$r]}"
  echo "[$(date +%T)] $r (origin-corrected)"
  .venv/bin/python -m evaluation.compute_subtraction --origin-correct \
    --config "$cfg" --checkpoint "checkpoints/$r/best.pth" \
    --out "evaluation/v2_metrics/${r}_subtraction_origin.json" \
    --per-chunk-out "$out" --seed 0
done
.venv/bin/python -m evaluation.paired_stats \
  --per-chunk 'evaluation/v2_metrics/per_chunk/*_per_chunk.json' \
  --out evaluation/v2_metrics/paired_stats.json
