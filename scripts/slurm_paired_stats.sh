#!/bin/bash
#SBATCH --job-name=paired
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
# Per-chunk stem-subtraction measurements for every GAN autoencoder run, then
# paired bootstrap CIs. Needs only MUSDB18 test + the HF checkpoints, so it does
# not wait on the FMA preprocessing.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNS="v2.0-continued v2.1-decmix v2.2-decmix-disc v3.0-baseline-d64 v3.1-decmix-disc-d64"
declare -A CFG=(
  [v2.0-continued]=configs/experiments/v2/v2.0_continued.yaml
  [v2.1-decmix]=configs/experiments/v2/v2.1_decmix.yaml
  [v2.2-decmix-disc]=configs/experiments/v2/v2.2_decmix_disc.yaml
  [v3.0-baseline-d64]=configs/experiments/v3/v3.0_baseline_d64.yaml
  [v3.1-decmix-disc-d64]=configs/experiments/v3/v3.1_decmix_disc_d64.yaml )
mkdir -p evaluation/v2_metrics/per_chunk
for r in $RUNS; do
  out=evaluation/v2_metrics/per_chunk/${r}_per_chunk.json
  [ -f "$out" ] && { echo "[skip] $r"; continue; }
  if [ ! -f "checkpoints/$r/best.pth" ]; then
    mkdir -p "checkpoints/$r"
    .venv/bin/python -c "
from huggingface_hub import hf_hub_download; import shutil, sys
p = hf_hub_download('SoMa25/mixing-equivariant-ae-checkpoints', f'musicgen/{sys.argv[1]}/best.pth')
shutil.copy(p, f'checkpoints/{sys.argv[1]}/best.pth')" "$r"
  fi
  cfg="checkpoints/$r/config.yaml"; [ -f "$cfg" ] || cfg="${CFG[$r]}"
  echo "[$(date +%T)] $r"
  .venv/bin/python -m evaluation.compute_subtraction \
    --config "$cfg" --checkpoint "checkpoints/$r/best.pth" \
    --out "evaluation/v2_metrics/${r}_subtraction_recheck.json" \
    --per-chunk-out "$out" --seed 0
done
.venv/bin/python -m evaluation.paired_stats \
  --per-chunk 'evaluation/v2_metrics/per_chunk/*_per_chunk.json' \
  --out evaluation/v2_metrics/paired_stats.json
