#!/bin/bash
#SBATCH --job-name=codecorig
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-%x-%j.out
# Origin recentering on third-party autoencoders that we did not train:
# DAC-44k and EnCodec-24k (no linearity objective at all), and the public
# Lin-CAE family (Torres et al.), including their retrained Music2Latent.
#
# Raw and +f(0) latent subtraction, per-chunk records, on:
#   - MUSDB18-HQ test  (all models; this is Lin-CAE's own test set)
#   - MoisesDB held-out corpus (DAC and EnCodec only: MoisesDB is training
#     data for the Lin-CAE family, and MUSDB is training data for DAC)
# Units are identical to the GAN and M2L runs (same seed, same track order),
# so evaluation.paired_stats / origin_effect apply unchanged. Resumable.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # weights were prefetched on the login node
PY=.venv/bin/python
MOISES="${1:-dataset/moisesdb_musdbform}"

run() {  # run <model> <corpus-dir> <outdir> <mode>
  local m=$1 corpus=$2 out=$3 mode=$4 flag="" suf=""
  [ "$mode" = origin ] && { flag="--origin-correct"; suf="+origin"; }
  mkdir -p "$out/per_chunk"
  [ -f "$out/per_chunk/${m}${suf}_per_chunk.json" ] && { echo "skip $m $mode ($out)"; return; }
  echo "=== $m $mode -> $out"
  $PY -m evaluation.codec_run_subtraction --model "$m" $flag --musdb-dir "$corpus" \
     --out "$out/${m}_subtraction${suf/+/_}.json" \
     --per-chunk-out "$out/per_chunk/${m}${suf}_per_chunk.json" --seed 0 --batch-size 16
}

OUT_MUSDB=evaluation/v2_metrics/codecs
OUT_MOISES=evaluation/holdout_moisesdb/codecs
for m in dac44k encodec24k lincae lincae-m2l lincae2; do
  for mode in raw origin; do run "$m" dataset/musdb18/test "$OUT_MUSDB" "$mode"; done
done
for m in dac44k encodec24k; do
  for mode in raw origin; do run "$m" "$MOISES" "$OUT_MOISES" "$mode"; done
done

# Per-model raw vs recentered statistics (track-cluster bootstrap). The MUSDB
# run also pulls in the GAN / M2L / v1.1 per-chunk records so one file holds
# every model. --key dd is the phase-cancelled reading for stochastic decoders.
for key in sub dd; do
  $PY -m evaluation.paired_stats --key $key \
      --per-chunk "evaluation/v2_metrics/per_chunk/*_per_chunk.json" "$OUT_MUSDB/per_chunk/*_per_chunk.json" \
      --out "$OUT_MUSDB/paired_stats_${key}.json" | tee "$OUT_MUSDB/paired_stats_${key}.txt"
done
$PY -m evaluation.origin_effect --per-chunk "$OUT_MUSDB/per_chunk/*_per_chunk.json" \
    --out "$OUT_MUSDB/origin_effect.json" | tee "$OUT_MUSDB/origin_effect.txt"
for key in sub dd; do
  $PY -m evaluation.paired_stats --key $key --activity evaluation/holdout_moisesdb/activity.json \
      --per-chunk "$OUT_MOISES/per_chunk/*_per_chunk.json" \
      --out "$OUT_MOISES/paired_stats_${key}.json" | tee "$OUT_MOISES/paired_stats_${key}.txt"
done
echo done
