#!/bin/bash
#SBATCH --job-name=fig1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
# Fig. 1 protocol comparison with identical mixing pairs in both protocols and
# an isolated decoder-noise stream (reviewer audit: pair sampling and noise
# previously shared the global RNG).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export MUSICGEN_M2L_PUBLISHED=checkpoints/m2l/music2latent.pt
export MUSICGEN_M2L_CHECKPOINT=checkpoints/m2l/mix_phase2_decmix_consmix_ema.pt
.venv/bin/python -m scripts._diag_old_vs_new_eval --max-batches 30 2>&1 | tee evaluation/v2_metrics/_diag_old_vs_new_eval_fixedpairs.log
