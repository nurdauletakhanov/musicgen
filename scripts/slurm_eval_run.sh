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
# Evaluate the FINAL checkpoint, never a selected one. The trainer writes
# step_25000.pth only after a final validation pass; when that pass was cut
# short, latest.pth already holds the same step-25000 weights, so it is the
# correct fallback and still involves no selection.
CKPT_PICK() { [ -f "checkpoints/$1/step_25000.pth" ] && echo step_25000.pth || echo latest.pth; }
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
RUN="${1:?run name under checkpoints/}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "checkpoints/$RUN/best.pth" ] || { echo "no checkpoints/$RUN/best.pth"; exit 1; }
CKPT=$(CKPT_PICK "$RUN")
# Do not call a checkpoint "final" without looking: latest.pth is only the
# final-step weights if it actually sits at the last step.
STEP=$(.venv/bin/python -c "
import torch,sys
c=torch.load('checkpoints/$RUN/$CKPT', map_location='cpu', weights_only=False)
print(c.get('global_step','?'))" 2>/dev/null || echo "?")
echo "evaluating checkpoints/$RUN/$CKPT at step $STEP"
if [ "$CKPT" = latest.pth ] && [ "$STEP" != 25000 ]; then
  echo "REFUSING: latest.pth is at step $STEP, not the final 25000, so this is"
  echo "not a selection-free final checkpoint. Finish the run first."
  exit 1
fi
[ -f checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt ] || { echo "CLAP checkpoint missing"; exit 1; }
.venv/bin/python -m scripts.run_v2_mixing_metrics --only "$RUN" --checkpoint-name "$CKPT"
.venv/bin/python -m scripts.run_v2_fad --only "$RUN" --checkpoint-name "$CKPT"
.venv/bin/python -m scripts.run_subtraction --only "$RUN" --checkpoint-name "$CKPT"
ls -la evaluation/v2_metrics/"$RUN"_*.json
