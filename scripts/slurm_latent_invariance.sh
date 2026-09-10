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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BS="${BATCH_SIZE:-16}"

# The gpu partition is shared and a card can be allocated with no free memory
# left on it. Fail loudly and early rather than after a partial sweep.
$PY - <<'CHK'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible")
free, total = torch.cuda.mem_get_info()
print(f"GPU free {free/2**30:.1f} GiB of {total/2**30:.1f} GiB")
if free < 4 * 2**30:
    sys.exit("less than 4 GiB free on the allocated GPU; requeue")
CHK
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
OUT=evaluation/holdout_moisesdb/latent_decomposition
mkdir -p "$OUT/per_chunk"
for r in "${RUNS[@]}"; do
  [ -f "$OUT/${r}_mixing.json" ] && { echo "skip $r"; continue; }
  echo "=== held-out: $r ==="
  $PY -m evaluation.compute_mixing_metrics \
      --config "$(cfg_for "$r")" --checkpoint "checkpoints/$r/best.pth" \
      --chunks-dir chunks-holdout --val-split test \
      --out "$OUT/${r}_mixing.json" \
      --per-unit-out "$OUT/per_chunk/${r}_mixing_per_unit.json" --alpha 0.5 --batch-size "$BS"
done

echo "=== decomposition and waveform cost ==="
$PY - <<'PY'
import json, os
RUNS=["v2.0-continued","v2.1-decmix","v2.2-decmix-disc","v3.0-baseline-d64","v3.1-decmix-disc-d64"]
for tag, d in (("held-out MoisesDB","evaluation/holdout_moisesdb/latent_decomposition"),
               ("MUSDB-era test set","evaluation/v2_metrics/latent_decomposition")):
    print(f"\n--- {tag}")
    print(f"  {'run':22s} {'N/D':>9s} {'S/D':>8s} {'O/D':>8s} {'N/S':>7s} {'N/O':>7s} {'gt(z_bar)':>10s} {'gt(direct)':>11s} {'cost':>6s}")
    for r in RUNS:
        p=os.path.join(d, f"{r}_mixing.json")
        if not os.path.exists(p): continue
        j=json.load(open(p)); m=j["metrics"]; g=lambda k: m.get(k,{}).get("all", float("nan"))
        print(f"  {r:22s} {g('lat_N'):9.5f} {g('lat_S'):8.4f} {g('lat_O'):8.4f} "
              f"{g('l_lat_span'):7.4f} {g('l_lat_centered'):7.4f} "
              f"{g('sdr_lin_gt'):10.2f} {g('sdr_direct_gt'):11.2f} {g('sdr_interp_cost'):6.2f}")
PY
echo done
