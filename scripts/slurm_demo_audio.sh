#!/bin/bash
#SBATCH --job-name=demo-audio
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
# Generate the audio-examples page assets (docs/audio + docs/manifest.json).
#
# Usage:
#   sbatch scripts/slurm_demo_audio.sh
#   bash   scripts/slurm_demo_audio.sh          # run directly on a GPU node
#
# Needs:
#   - dataset/musdb18/test/          (MUSDB18-HQ test split, raw WAV stems)
#   - checkpoints/<run>/best.pth     for v2.0/v2.1/v3.0/v3.1 — auto-downloaded
#     from the HF repo if missing (export HF_TOKEN=... while it is private)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "host=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

if [ ! -x .venv/bin/pip ]; then
  # the cluster's python has no ensurepip, so bootstrap pip by hand
  python3 -m venv --without-pip .venv
  curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
  # node driver 570 supports CUDA <= 12.8; pip's default torch wheel won't init
  .venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/pip install -q -r requirements.txt
fi

[ -d dataset/musdb18/test ] || { echo "ERROR: dataset/musdb18/test missing"; exit 1; }

# Fetch any missing checkpoints from the HF weights repo.
HF_REPO="${MUSICGEN_HF_REPO:-SoMa25/mixing-equivariant-ae-checkpoints}"
for run in v2.0-continued v2.1-decmix v3.0-baseline-d64 v3.1-decmix-disc-d64; do
  if [ ! -f "checkpoints/$run/best.pth" ]; then
    echo "fetching $run/best.pth from $HF_REPO"
    .venv/bin/python - "$run" "$HF_REPO" <<'PYEOF'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
run, repo = sys.argv[1], sys.argv[2]
p = hf_hub_download(repo_id=repo, filename=f"musicgen/{run}/best.pth")
dst = Path("checkpoints") / run / "best.pth"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(p, dst)
print(f"  -> {dst}")
PYEOF
  fi
done

.venv/bin/python -m scripts.generate_demo_audio "$@"

echo "end=$(date)"
echo "now: git add docs && git commit -m 'Add audio examples' && git push"
echo "then enable GitHub Pages: repo Settings -> Pages -> main, /docs"
