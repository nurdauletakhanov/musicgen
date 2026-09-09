#!/bin/bash
#SBATCH --job-name=mixunit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%x-%j.out
# Re-run the convex-mixing eval on the MUSDB-era test split, this time writing
# identified per-unit records so the mixing contrasts get paired intervals.
#
# Written to a SEPARATE directory: the published *_mixing.json files stay put
# until the two runs have been compared. Tables and statistics must end up
# reading the same records, which is the point of the exercise.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=.venv/bin/python
OUT=evaluation/v2_metrics/mixing_v2
mkdir -p "$OUT/per_chunk"

RUNS=(v2.0-continued v2.1-decmix v2.2-decmix-disc v3.0-baseline-d64 v3.1-decmix-disc-d64)
cfg_for() {
  case "$1" in
    v3.0-baseline-d64)     echo configs/experiments/v3/v3.0_baseline_d64.yaml ;;
    v3.1-decmix-disc-d64)  echo configs/experiments/v3/v3.1_decmix_disc_d64.yaml ;;
    v2.0-continued)        echo configs/experiments/v2/v2.0_continued.yaml ;;
    v2.1-decmix)           echo configs/experiments/v2/v2.1_decmix.yaml ;;
    v2.2-decmix-disc)      echo configs/experiments/v2/v2.2_decmix_disc.yaml ;;
  esac
}

for r in "${RUNS[@]}"; do
  out="$OUT/${r}_mixing.json"
  [ -f "$out" ] && { echo "skip $r (exists)"; continue; }
  echo "=== mixing (per-unit): $r ==="
  $PY -m evaluation.compute_mixing_metrics \
      --config "$(cfg_for "$r")" \
      --checkpoint "checkpoints/$r/best.pth" \
      --out "$out" \
      --per-unit-out "$OUT/per_chunk/${r}_mixing_per_unit.json" \
      --alpha 0.5 --seed 0
done

$PY -m evaluation.mixing_stats \
    --per-unit "$OUT/per_chunk/*_mixing_per_unit.json" \
    --out "$OUT/mixing_stats.json" | tee "$OUT/mixing_stats.txt"

echo "=== published vs re-run summary (sdr_lin_gt, all) ==="
$PY - <<'PY'
import json, os
for r in ["v2.0-continued","v2.1-decmix","v2.2-decmix-disc","v3.0-baseline-d64","v3.1-decmix-disc-d64"]:
    a=f"evaluation/v2_metrics/{r}_mixing.json"; b=f"evaluation/v2_metrics/mixing_v2/{r}_mixing.json"
    if not (os.path.exists(a) and os.path.exists(b)): continue
    ga=json.load(open(a))["metrics"]["sdr_lin_gt"]["all"]; gb=json.load(open(b))["metrics"]["sdr_lin_gt"]["all"]
    print(f"  {r:24s} published {ga:+.4f}  rerun {gb:+.4f}  delta {gb-ga:+.4f}")
PY
echo "done. results in $OUT"
