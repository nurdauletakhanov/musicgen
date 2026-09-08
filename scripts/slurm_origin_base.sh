#!/bin/bash
#SBATCH --job-name=originb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
# Origin correction on the base models: the recon-only v1.1 autoencoder and
# Music2Latent (published, and our Phase-2 fine-tune). Raw and +f(0), per-chunk.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PC=evaluation/v2_metrics/per_chunk; mkdir -p "$PC" checkpoints/v1.1 checkpoints/m2l
.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download; import shutil, os
for repo, fn, dst in [("SoMa25/mixing-equivariant-ae-checkpoints","musicgen/v1.1/best.pth","checkpoints/v1.1/best.pth"),
                      ("SoMa25/mixing-equivariant-ae-checkpoints","m2l/mix_phase2_decmix_consmix_ema.pt","checkpoints/m2l/mix_phase2_decmix_consmix_ema.pt"),
                      ("SonyCSLParis/music2latent","music2latent.pt","checkpoints/m2l/music2latent.pt")]:
    if not os.path.exists(dst): shutil.copy(hf_hub_download(repo, fn), dst); print("fetched", dst)
PY
for mode in raw origin; do
  flag=""; suf=""; [ $mode = origin ] && { flag="--origin-correct"; suf="+origin"; }
  [ -f "$PC/v1.1${suf}_per_chunk.json" ] || .venv/bin/python -m evaluation.compute_subtraction $flag \
     --config configs/experiments/v1/v1.1.yaml --checkpoint checkpoints/v1.1/best.pth \
     --out "evaluation/v2_metrics/v1.1_subtraction${suf/+/_}.json" --per-chunk-out "$PC/v1.1${suf}_per_chunk.json" --seed 0
  [ -f "$PC/m2l_phase0_ema${suf}_per_chunk.json" ] || .venv/bin/python -m evaluation.m2l_run_subtraction $flag \
     --m2l-checkpoint checkpoints/m2l/music2latent.pt \
     --out "evaluation/v2_metrics/m2l_phase0_ema_subtraction${suf/+/_}.json" --per-chunk-out "$PC/m2l_phase0_ema${suf}_per_chunk.json" --seed 0
  [ -f "$PC/m2l_phase2_ema${suf}_per_chunk.json" ] || .venv/bin/python -m evaluation.m2l_run_subtraction $flag \
     --m2l-checkpoint checkpoints/m2l/mix_phase2_decmix_consmix_ema.pt \
     --out "evaluation/v2_metrics/m2l_phase2_ema_subtraction${suf/+/_}.json" --per-chunk-out "$PC/m2l_phase2_ema${suf}_per_chunk.json" --seed 0
done
.venv/bin/python -m evaluation.paired_stats --per-chunk "$PC/*_per_chunk.json" --out evaluation/v2_metrics/paired_stats.json
