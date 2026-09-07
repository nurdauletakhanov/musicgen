#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
# Full paper evaluation for one run: mixing metrics (full test set), FAD, stem
# subtraction -> evaluation/v2_metrics/<run>_{mixing,fad,subtraction}.json.
#   sbatch --job-name=ev-v21s2 scripts/slurm_eval_run.sh v2.1-decmix-s2
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
RUN="${1:?run name under checkpoints/}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "checkpoints/$RUN/best.pth" ] || { echo "no checkpoints/$RUN/best.pth"; exit 1; }
[ -f checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt ] || { echo "CLAP checkpoint missing"; exit 1; }
.venv/bin/python -m scripts.run_v2_mixing_metrics --only "$RUN"
.venv/bin/python -m scripts.run_v2_fad --only "$RUN"
.venv/bin/python -m scripts.run_subtraction --only "$RUN"
ls -la evaluation/v2_metrics/"$RUN"_*.json
