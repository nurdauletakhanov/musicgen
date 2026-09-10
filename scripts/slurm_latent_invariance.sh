#!/bin/bash
#SBATCH --job-name=latinv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
# Does the ell_lat ranking survive a translation-invariant normalization?
#
# ell_lat = ||zbar - f(xbar)||^2 / ||f(xbar)||^2 is NOT invariant to relabelling
# the latent as f(x)+b, g(z-b), which leaves reconstruction and decoded convex
# mixing pointwise unchanged. The numerator is invariant (mixing coefficients
# sum to one); the denominator is not, so a large offset shrinks the reported
# error for free. This re-runs the mixing eval recording, alongside it:
#   l_lat_abs      unnormalized residual per latent element (translation-inv.)
#   l_lat_span     residual / ||f(x1)-f(x2)||^2   (translation- and scale-inv.)
#   l_lat_centered residual / ||f(xbar)-f(0)||^2  (same recentering as the
#                                                  subtraction analysis)
# and ||f(0)|| per model, which is what inflates the old denominator.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=.venv/bin/python
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

# 1. Held-out corpus first: 14k chunks, minutes per model, and it answers the
#    question on data that took no part in training or selection.
OUT=evaluation/holdout_moisesdb/latent_invariance
mkdir -p "$OUT/per_chunk"
for r in "${RUNS[@]}"; do
  [ -f "$OUT/${r}_mixing.json" ] && { echo "skip $r"; continue; }
  echo "=== held-out: $r ==="
  $PY -m evaluation.compute_mixing_metrics \
      --config "$(cfg_for "$r")" --checkpoint "checkpoints/$r/best.pth" \
      --chunks-dir chunks-holdout --val-split test \
      --out "$OUT/${r}_mixing.json" \
      --per-unit-out "$OUT/per_chunk/${r}_mixing_per_unit.json" --alpha 0.5
done

# 2. The paper's test set, which is where the published ell_lat numbers come
#    from. Balanced subsample keeps this to minutes rather than an hour a model.
OUT2=evaluation/v2_metrics/latent_invariance
mkdir -p "$OUT2"
for r in "${RUNS[@]}"; do
  [ -f "$OUT2/${r}_mixing.json" ] && { echo "skip $r"; continue; }
  echo "=== test set (balanced 400/source): $r ==="
  $PY -m evaluation.compute_mixing_metrics \
      --config "$(cfg_for "$r")" --checkpoint "checkpoints/$r/best.pth" \
      --out "$OUT2/${r}_mixing.json" --per-source 400 --alpha 0.5
done

echo "=== rankings under each normalization ==="
$PY - <<'PY'
import json, os
RUNS=["v2.0-continued","v2.1-decmix","v2.2-decmix-disc","v3.0-baseline-d64","v3.1-decmix-disc-d64"]
for tag, d in (("held-out MoisesDB","evaluation/holdout_moisesdb/latent_invariance"),
               ("MUSDB-era test set","evaluation/v2_metrics/latent_invariance")):
    print(f"\n--- {tag}")
    print(f"  {'run':24s} {'||f(0)||':>9s} {'l_lat':>9s} {'abs':>10s} {'span':>9s} {'centered':>9s}")
    for r in RUNS:
        p=os.path.join(d, f"{r}_mixing.json")
        if not os.path.exists(p): continue
        j=json.load(open(p)); m=j["metrics"]; g=lambda k: m.get(k,{}).get("all", float("nan"))
        print(f"  {r:24s} {j.get('f0_norm',float('nan')):9.2f} {g('l_lat'):9.4f} "
              f"{g('l_lat_abs'):10.4f} {g('l_lat_span'):9.4f} {g('l_lat_centered'):9.4f}")
PY
echo done
