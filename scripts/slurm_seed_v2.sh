#!/bin/bash
#SBATCH --job-name=seed
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=slurm-%x-%j.out
# Independent re-run ("seed") of a v2.x fine-tune config under a new run name.
#   sbatch --job-name=v21s2 scripts/slurm_seed_v2.sh configs/experiments/v2/v2.1_decmix.yaml s2
# Warm-starts from checkpoints/v1.1/best.pth (fetched from HF if absent), like the
# paper's v2.x runs. Resumes from latest.pth and resubmits itself across walltime.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CONFIG="${1:?config.yaml}"; SUFFIX="${2:-s2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d chunks-44k-1s ] && [ -f .chunks_ready ] || { echo "chunks-44k-1s not ready"; exit 1; }
if [ ! -f checkpoints/v1.1/best.pth ]; then
  mkdir -p checkpoints/v1.1
  .venv/bin/python -c "
from huggingface_hub import hf_hub_download; import shutil
p=hf_hub_download('SoMa25/mixing-equivariant-ae-checkpoints','musicgen/v1.1/best.pth'); shutil.copy(p,'checkpoints/v1.1/best.pth')"
fi
BASE="$(sed -n 's/^name:[[:space:]]*//p' "$CONFIG" | head -1)"; RUN="${BASE}-${SUFFIX}"
CFG="configs/experiments/generated_${RUN}.yaml"; sed "s/^name:.*/name: ${RUN}/" "$CONFIG" > "$CFG"
DIR="checkpoints/${RUN}"
[ -f "$DIR/step_0025000.pth" ] || [ -f "$DIR/step_25000.pth" ] && { echo "$RUN finished"; exit 0; }
if [ -f "$DIR/latest.pth" ]; then ARGS=(--resume "$DIR/latest.pth"); else ARGS=(--warm-start checkpoints/v1.1/best.pth); fi
set +e; timeout 80400 .venv/bin/python -m training.train --config "$CFG" "${ARGS[@]}"; ST=$?; set -e
if [ $ST -eq 124 ]; then [ -n "${SLURM_JOB_ID:-}" ] && sbatch --job-name="$SLURM_JOB_NAME" "$0" "$CONFIG" "$SUFFIX"; exit 0; fi
exit $ST
