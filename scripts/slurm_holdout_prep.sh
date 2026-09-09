#!/bin/bash
#SBATCH --job-name=hprep
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%x-%j.out
# Turn the raw MoisesDB download into the two things the held-out evaluation
# needs: a MUSDB18-layout corpus for subtraction, and 1-second chunks of the
# mixtures for the convex-mixing probe. CPU work only.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=.venv/bin/python
SRC="${1:-$HOME/datasets/moisesdb/moisesdb}"
CORPUS=dataset/moisesdb_musdbform
CHUNKS=chunks-holdout

[ -d "$SRC" ] || { echo "source not found: $SRC"; exit 1; }
echo "source: $SRC ($(ls -1 "$SRC" | wc -l) track dirs)"

if [ ! -f "$CORPUS/manifest.json" ]; then
  echo "=== converting to MUSDB18 layout ==="
  $PY -m scripts.prepare_moisesdb --src "$SRC" --out "$CORPUS"
else
  echo "corpus already built: $CORPUS"
fi

if [ ! -f "$CHUNKS/index.json" ]; then
  echo "=== chunking mixtures for the mixing probe ==="
  $PY -m scripts.chunk_holdout --src "$CORPUS" --out "$CHUNKS" \
      --seconds 1.0 --max-chunks-per-track 60
else
  echo "chunks already built: $CHUNKS"
fi

echo "=== additivity check on a sample of the built corpus ==="
$PY - <<'PY'
import glob, numpy as np, soundfile as sf, random
dirs = sorted(glob.glob("dataset/moisesdb_musdbform/*/"))
random.Random(0).shuffle(dirs)
worst = 0.0
for d in dirs[:8]:
    st = {s: sf.read(f"{d}{s}.wav", dtype="float32", always_2d=True)[0]
          for s in ("mixture", "drums", "bass", "vocals", "other")}
    n = min(len(v) for v in st.values())
    resid_err = max(
        float(np.abs(st["mixture"][:n] - st[t][:n]
                     - sum(st[o][:n] for o in ("drums","bass","vocals","other") if o != t)).max())
        for t in ("drums","bass","vocals","other"))
    worst = max(worst, resid_err)
print(f"max |mixture - target - residual| over {min(8,len(dirs))} tracks: {worst:.3e}")
print("(must be at float32 rounding level; the subtraction test depends on it)")
PY
echo "done"
