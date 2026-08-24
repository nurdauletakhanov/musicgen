#!/bin/bash
#SBATCH --job-name=rerun
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=slurm-%x-%j.out
# Independent re-run ("second seed") of a training config, for the paper's
# seed-variance question. Training has no explicit seed, so a re-run under a
# new output name IS the second draw.
#
# Usage:
#   sbatch --job-name=v30r2 scripts/slurm_train_rerun.sh configs/experiments/v3/v3.0_baseline_d64.yaml
#   sbatch --job-name=v31r2 scripts/slurm_train_rerun.sh configs/experiments/v3/v3.1_decmix_disc_d64.yaml
#
# A v3 run is ~35 h; walltime is 23 h. The script resumes from latest.pth and
# resubmits itself until step_250000.pth exists, so one sbatch is enough.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

CONFIG="${1:?usage: slurm_train_rerun.sh <config.yaml> [suffix]}"
SUFFIX="${2:-run2}"

echo "host=$(hostname) config=$CONFIG suffix=$SUFFIX start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

if [ ! -x .venv/bin/pip ]; then
  python3 -m venv --without-pip .venv
  curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
  .venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/pip install -q -r requirements.txt
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
  .venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "ERROR: CUDA not available on $(hostname); refusing to train on CPU"; exit 1; }
fi

[ -d chunks-44k-1s ] || { echo "ERROR: chunks-44k-1s/ missing — run data.preprocess first"; exit 1; }

# Derive the re-run config: same recipe, new name -> new checkpoints/<name> dir.
BASE_NAME="$(sed -n 's/^name:[[:space:]]*//p' "$CONFIG" | head -1)"
RUN="${BASE_NAME}-${SUFFIX}"
RERUN_CFG="configs/experiments/generated_${RUN}.yaml"
sed "s/^name:.*/name: ${RUN}/" "$CONFIG" > "$RERUN_CFG"
CKPT_DIR="checkpoints/${RUN}"

if [ -f "$CKPT_DIR/step_250000.pth" ]; then
  echo "$RUN already finished"; exit 0
fi

RESUME=()
[ -f "$CKPT_DIR/latest.pth" ] && RESUME=(--resume "$CKPT_DIR/latest.pth")

# Stop 40 min before walltime so the checkpoint write is never cut off,
# then resubmit this same script to continue from latest.pth.
set +e
timeout 80400 .venv/bin/python -m training.train --config "$RERUN_CFG" "${RESUME[@]}"
STATUS=$?
set -e

if [ "$STATUS" -eq 124 ]; then
  echo "walltime slice used; resubmitting to continue $RUN"
  [ -n "${SLURM_JOB_ID:-}" ] && sbatch --job-name="${SLURM_JOB_NAME}" "$0" "$CONFIG" "$SUFFIX"
  exit 0
fi
echo "end=$(date) status=$STATUS"
exit "$STATUS"
