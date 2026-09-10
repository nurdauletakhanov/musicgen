#!/bin/bash
#SBATCH --job-name=holdout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
# Held-out evaluation on MoisesDB, exactly as pre-registered in
# research/paper/icassp2027/HOLDOUT_PROTOCOL.md.
#
# Frozen weights, corpus that took no part in training or in checkpoint
# selection. Runs the five GAN checkpoints on identical units:
#   - stem subtraction, raw and origin-recentered, with comparators
#   - convex mixing metrics with per-unit records (for the dyadic bootstrap)
#
# Every checkpoint hash is verified against the protocol before anything runs;
# a mismatch aborts, because the whole point is that the weights did not move.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=.venv/bin/python
CORPUS="${1:-dataset/moisesdb_musdbform}"  # built by scripts/slurm_holdout_prep.sh
OUT=evaluation/holdout_moisesdb
mkdir -p "$OUT/per_chunk"

[ -d "$CORPUS" ] || { echo "corpus not found: $CORPUS (run scripts.prepare_moisesdb first)"; exit 1; }
echo "corpus: $CORPUS ($(ls -1 "$CORPUS" | grep -vc manifest.json) tracks)"

# --- checkpoint integrity, from HOLDOUT_PROTOCOL.md -------------------------
declare -A SHA=(
  [v2.0-continued]=92718bfe15f325b417fda7fb719c6ab6d60c424f1230101380ac088899460bd4
  [v2.1-decmix]=f47c8dc58c6f44aeae4df8d9c693db13c65552d0c028815e38635be126b1d891
  [v2.2-decmix-disc]=3770e23fdd8d598ec3a098347aa09bdba3c9cd31191a76c1b1b9c0f0126dbb54
  [v3.0-baseline-d64]=5668eca13abc0ee78d641892ecdecd06533c9dd745002271e354c101159dfbcc
  [v3.1-decmix-disc-d64]=70df0436b57a2914ae6e0f3b3450d8a53a93aff39eba96d11f6b3624b3e1cb36
)
RUNS=(v2.0-continued v2.1-decmix v2.2-decmix-disc v3.0-baseline-d64 v3.1-decmix-disc-d64)
for r in "${RUNS[@]}"; do
  f="checkpoints/$r/best.pth"
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$got" = "${SHA[$r]}" ] || { echo "HASH MISMATCH $r: $got != ${SHA[$r]}"; exit 1; }
done
echo "all five checkpoints match the pre-registered hashes"

cfg_for() {  # run name -> its training config
  case "$1" in
    v3.*) echo "configs/experiments/v3/$(echo "$1" | sed 's/v3\.0-baseline-d64/v3.0_baseline_d64/;s/v3\.1-decmix-disc-d64/v3.1_decmix_disc_d64/').yaml" ;;
    v2.0-continued)    echo configs/experiments/v2/v2.0_continued.yaml ;;
    v2.1-decmix)       echo configs/experiments/v2/v2.1_decmix.yaml ;;
    v2.2-decmix-disc)  echo configs/experiments/v2/v2.2_decmix_disc.yaml ;;
  esac
}

# --- 1. stem subtraction, raw and recentered -------------------------------
for r in "${RUNS[@]}"; do
  for mode in raw origin; do
    tag="$r"; extra=()
    if [ "$mode" = origin ]; then tag="$r+origin"; extra=(--origin-correct); fi
    out="$OUT/${tag}_subtraction.json"
    [ -f "$out" ] && { echo "skip $tag (exists)"; continue; }
    echo "=== subtraction: $tag ==="
    $PY -m evaluation.compute_subtraction \
        --config "$(cfg_for "$r")" \
        --checkpoint "checkpoints/$r/best.pth" \
        --musdb-dir "$CORPUS" \
        --out "$out" \
        --per-chunk-out "$OUT/per_chunk/${tag}_per_chunk.json" \
        "${extra[@]}"
  done
done

# --- 2. convex mixing, with per-unit records for the dyadic bootstrap ------
# Mixing pairs come from the held-out corpus too, so no FMA is involved.
CHUNKS=chunks-holdout
if [ ! -f "$CHUNKS/index.json" ]; then
  echo "=== chunking held-out mixtures for the mixing probe ==="
  $PY -m scripts.chunk_holdout --src "$CORPUS" --out "$CHUNKS" \
      --seconds 1.0 --max-chunks-per-track 60
fi
for r in "${RUNS[@]}"; do
  out="$OUT/${r}_mixing.json"
  [ -f "$out" ] && { echo "skip mixing $r (exists)"; continue; }
  echo "=== mixing: $r ==="
  $PY -m evaluation.compute_mixing_metrics \
      --config "$(cfg_for "$r")" \
      --checkpoint "checkpoints/$r/best.pth" \
      --chunks-dir "$CHUNKS" --val-split test \
      --out "$out" \
      --per-unit-out "$OUT/per_chunk/${r}_mixing_per_unit.json" \
      --alpha 0.5
done

# --- 3. the pre-registered silent-target rule ------------------------------
# Recovered from the corpus audio using each record's sample offset; no model,
# no re-run. Thresholds are those fixed in HOLDOUT_PROTOCOL.md section 4.
if [ ! -f "$OUT/activity.json" ]; then
  $PY -m evaluation.holdout_activity \
      --per-chunk "$OUT/per_chunk/v2.0-continued_per_chunk.json" \
      --corpus "$CORPUS" --out "$OUT/activity.json" | tee "$OUT/activity.txt"
fi

# --- 4. statistics: pre-registered (filtered) and unfiltered ---------------
$PY -m evaluation.origin_effect \
    --per-chunk "$OUT/per_chunk/*_per_chunk.json" \
    --activity "$OUT/activity.json" \
    --out "$OUT/origin_effect.json" | tee "$OUT/origin_effect.txt"
$PY -m evaluation.paired_stats \
    --per-chunk "$OUT/per_chunk/*_per_chunk.json" \
    --activity "$OUT/activity.json" \
    --out "$OUT/paired_stats.json" | tee "$OUT/paired_stats.txt"

echo "=== unfiltered, for comparison ==="
$PY -m evaluation.paired_stats \
    --per-chunk "$OUT/per_chunk/*_per_chunk.json" \
    --out "$OUT/paired_stats_unfiltered.json" | tee "$OUT/paired_stats_unfiltered.txt"
$PY -m evaluation.origin_effect \
    --per-chunk "$OUT/per_chunk/*_per_chunk.json" \
    --out "$OUT/origin_effect_unfiltered.json" | tee "$OUT/origin_effect_unfiltered.txt"

# The mixing half needs its own statistics: each unit joins two recordings.
$PY -m evaluation.mixing_stats \
    --per-unit "$OUT/per_chunk/*_mixing_per_unit.json" \
    --out "$OUT/mixing_stats.json" | tee "$OUT/mixing_stats.txt"

echo "done. results in $OUT"
