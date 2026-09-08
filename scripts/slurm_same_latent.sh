#!/bin/bash
#SBATCH --job-name=samez
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p checkpoints/m2l
[ -f checkpoints/m2l/music2latent.pt ] || .venv/bin/python -c "
from huggingface_hub import hf_hub_download; import shutil
shutil.copy(hf_hub_download('SonyCSLParis/music2latent', 'music2latent.pt'), 'checkpoints/m2l/music2latent.pt')"
.venv/bin/python -m scripts._diag_same_latent --checkpoint checkpoints/m2l/music2latent.pt --max-batches 30
