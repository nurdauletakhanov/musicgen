# Music Autoencoder — v1 clean baseline

A wave-to-wave continuous-latent autoencoder for 44.1 kHz music, trained on a ~950-hour
mix of FMA-large, MAESTRO, and MUSDB18-HQ. Step-based training with an MSSTFTD + MPD
discriminator and multi-resolution STFT + mel reconstruction losses.

**v1 is a recon-only baseline.** No mixing losses, no stem-pair training, no alpha
sweeps — just clean reconstruction quality on a larger, more diverse training set than
prior iterations. Mixing-equivariance experiments become v2+, once the baseline FAD is
strong enough to afford the perceptual-quality cost of the extra supervision.

The pre-v1 lineage (STFT autoencoder, mixing/decode-mix experiments, M2L fine-tuning,
diffusion decoder) is preserved under `archive/pre-v1/` and its former configs under
`archive/pre-v1/configs-*/`.

---

## Architecture

33.5M parameters total, wave-to-wave, no transformer.

```
waveform [B, 1, 44100]  ── 1 s at 44.1 kHz ──>
  encoder (DAC-style)                    ──>  latent [B, 45, 128]   (5,760 floats, ~7.66× compression)
    4 stages, strides [4, 5, 7, 7] = 980
    channels       [64, 128, 256, 512, 512]
    dilated resblocks (1, 3, 9) per stage
    weight-norm Conv1d
  decoder (HiFi-GAN)                     ──>  waveform [B, 1, 44100]
    reversed strides [7, 7, 5, 4]
    channels       [512, 512, 256, 128, 64]
    ConvTranspose1d + MRF blocks (k=3,7 with dilations [1,3])
    weight-norm Conv1d
```

**Encoder** (`models/encoder.py`): strided 1-D convs downsample the waveform; each stage
is followed by a residual block with three parallel dilated Conv1d branches. The product
of encoder strides equals `44100 / num_segments = 980`, so the latent sequence is exactly
45 tokens of 128 dims.

**Decoder** (`models/decoder.py`): HiFi-GAN V1 recipe — ConvTranspose1d upsampling stages
interleaved with Multi-Receptive-Field (MRF) fusion blocks that average parallel
dilated-kernel residual stacks.

**Discriminator** (`models/discriminator.py`):
- **MSSTFTD** — Multi-Scale STFT discriminator at three FFT sizes
- **MPD** — Multi-Period discriminator at HiFi-GAN's default periods (2, 3, 5, 7, 11)
- `CombinedDiscriminator` runs both and concatenates their outputs

Adversarial + feature-matching losses are applied only after `disc_start_step = 10,000`
so the generator can first learn coarse reconstruction without disc pressure.

### Losses

Training loss is a weighted sum:

| Term | Weight | Description |
|---|---|---|
| Multi-resolution STFT | 1.0 | Magnitude log-L1 + spectral convergence at FFTs [256, 512, 1024, 2048] |
| Mel reconstruction | 1.0 | L1 on log-mel spectrogram |
| Latent L2 | 0.001 | Keeps latent magnitudes bounded |
| Adversarial (disc) | 0.5 | MSSTFTD + MPD, enabled after step 10,000 |
| Feature matching | 2.0 | L1 on intermediate disc activations |

No mixing loss, no decode-mix supervision, no stem-pair reconstruction for v1.

---

## Data

All three sources land in one unified chunks directory (`chunks-44k-1s/`) with a shared
`index.json`. The `source` tag on every entry (`fma`, `maestro`, `musdb`) is what lets
validation report per-domain SI-SDR without re-running on different directories.

**Chunks** are 1 s at 44.1 kHz = 44,100 samples, stored as `fp16` with the per-chunk peak
(for recovering absolute loudness) also saved. Peak-normalized to 0.95 on write so the
model always sees audio in [-0.95, 0.95]. Non-overlapping on a fixed grid.

**Train (96,264 tracks / 950.8 h total):**

| Source | Tracks | Chunks | Hours |
|---|---|---|---|
| FMA-large (training + validation splits) | 95,065 | 2,767,005 | 768.6 |
| MAESTRO v3 (training + validation) | 1,099 | 633,858 | 176.1 |
| MUSDB18-HQ train | 100 | 22,096 | 6.1 |

**Test (11,466 tracks / 113.7 h total):**

| Source | Tracks | Chunks | Hours |
|---|---|---|---|
| FMA-large (official test split) | 11,239 | 325,954 | 90.5 |
| MAESTRO test | 177 | 71,214 | 19.8 |
| MUSDB18-HQ test | 50 | 12,055 | 3.3 |

Test covers three distinct domains (real production / clean piano / diverse indie) so
SI-SDR, FAD, or any other eval can be reported both aggregate and per-source.

### Sources

- **FMA-large** (`dataset/fma_large/`, 93 GB MP3) — 106,574 Creative-Commons indie tracks,
  30 s each. Official 80/10/10 splits via `fma_metadata/tracks.csv`. 270 tracks skipped
  (0.25%) for known-bad data in the FMA archive (truncated stubs, silent files, corrupt
  frames). Loaded directly via `soundfile`/`libsndfile` 1.2+ native MP3 support (no
  ffmpeg dependency).
- **MAESTRO v3** (`dataset/maestro-v3.0.0/`) — 1,276 solo classical piano recordings.
  Source splits `{training, validation}` → our `train`, source `test` → our `test`.
- **MUSDB18-HQ** (`dataset/musdb18/`) — 150 rock/pop tracks with separate WAV stems.
  v1 uses only `mixture.wav`. Stem files remain on disk for v2+ mixing experiments.

---

## Training

Step-based schedule — robust to dataset size changes.

| Knob | Value |
|---|---|
| `max_steps` | 250,000 |
| `warmup_steps` | 3,000 (linear LR from 1e-3 × peak to peak) |
| LR schedule | Cosine to 10% of peak over remaining 247,000 steps |
| `batch_size` | 64 |
| Effective batch | 64 (no grad accum) |
| Optimizer | AdamW, betas (0.8, 0.99), wd 1e-3 |
| `learning_rate` (gen) | 3e-4 |
| `disc_lr` | 2e-4, betas (0.5, 0.9), wd 0 |
| `grad_clip` | 1.0 |
| `disc_start_step` | 10,000 |
| Precision | fp32 (bf16 AMP caused a step-4700 regression on the original v1 run) |
| `save_every_steps` | 10,000 |
| `log_every_steps` | 100 |

Total training examples seen: 64 × 250,000 = 16M, ≈ **4.7 passes** over the 3.42M unique
training chunks. At batch 64 fp32, per-step cost is roughly equivalent to the original
bf16 batch 128 plan, so wall-clock stays around **~35 hours**.

### Output layout

```
checkpoints/v1.1/
  config.yaml           # copy of the YAML used for this run
  train.log             # console tee (human-readable)
  train_log.jsonl       # one line per log-interval; machine-readable
  val_log.jsonl         # one line per val checkpoint (loss, MR-STFT, Mel, SI-SDR total + per-source)
  latest.pth            # most recent checkpoint (atomic-rename, safe to resume)
  best.pth              # lowest val loss so far
  step_250000.pth       # final checkpoint at end of training
  samples/
    step_0000010000/    # at each val checkpoint: listenable ref/recon .wav pairs
      00_musdb_ref.wav
      00_musdb_hat.wav
      01_maestro_ref.wav
      01_maestro_hat.wav
      ...
```

No TensorBoard, no HuggingFace Hub — the JSONL files are sufficient for plotting and
offline analysis, and `.wav` dumps let you audit reconstructions without running
additional inference.

---

## Quickstart

### 1. One-time preprocessing

```
python -m data.preprocess musdb
python -m data.preprocess maestro
python -m data.preprocess fma --workers 8
python -m scripts.reshuffle_fma_splits       # moves FMA's official test split into test/
```

Each preprocessor is resumable (checks existing `.pt` files). `--force` before the
subcommand (e.g. `python -m data.preprocess --force fma`) regenerates everything.

Outputs to `./chunks-44k-1s/{train,test}/` with a shared `index.json`.

### 2. Train

```
python -m training.train --config configs/experiments/v1/v1.1.yaml
```

Resume from any checkpoint:

```
python -m training.train --config configs/experiments/v1/v1.1.yaml \
    --resume ./checkpoints/v1.1/latest.pth
```

### 3. Listen

While the run is alive, the most recent `.wav` dump lives at the highest-numbered
`checkpoints/v1.1/samples/step_<N>/` directory. Open `00_musdb_ref.wav` and
`00_musdb_hat.wav` side-by-side to audit how the reconstruction sounds on MUSDB; same
for `maestro` and `fma` prefixes.

---

## Project layout

```
dataset/                       # raw data (all .gitignored)
  musdb18/{train,test}/<TrackName>/{mixture,drums,bass,other,vocals}.wav
  maestro-v3.0.0/<year>/<piece>.wav
  fma_large/<xxx>/<trackid>.mp3
  fma_metadata/tracks.csv      # provides official 80/10/10 split

chunks-44k-1s/                 # preprocessed (.gitignored)
  index.json                   # {"train": {key: entry}, "test": {key: entry}}
  train/<key>.pt               # {"x_wave": fp16 [N, 44100], "peak": fp32 [N]}
  test/<key>.pt

data/
  dataset.py                   # WaveformDataset + FileGroupedSampler + build_dataloaders
  preprocess.py                # unified CLI: musdb | maestro | fma

models/
  encoder.py                   # WaveEncoder
  decoder.py                   # WaveDecoder
  autoencoder.py               # glues encoder + decoder + reconstruction losses
  discriminator.py             # MSSTFTD + MPD + combined

training/
  trainer.py                   # Trainer class (step-based), ~400 lines
  train.py                     # CLI entry point
  config.py                    # YAML loader + build_model_config

configs/
  base.yaml                    # slim v1-compatible defaults
  experiments/v1/v1.1.yaml # this run

scripts/
  reshuffle_fma_splits.py      # one-time: move FMA test split from train/ to test/

archive/pre-v1/                # pre-v1 lineage, kept for reference
  code/{data,training,evaluation,scripts,utils}/...
  configs-{compression,compression-21x,diffusion,mixing}/
```

---

## Notes on the v1 → v2 boundary

The file `archive/pre-v1/` preserves everything from the v5–v17 experiments: old STFT
autoencoder, stempeg-based MUSDB preprocessing, `StemPairDataset`, decode-mix and
latent-mix losses, alpha-sweep evaluation, TensorBoard and HF Hub integration, M2L
fine-tuning, the Vocos decoder, and the diffusion training pipeline. None of that is
imported by v1 code — v1 only shares the architecture classes
(`models/{encoder,decoder,autoencoder,discriminator}.py`).

v2's plan is:
1. Demonstrate v1 achieves solid reconstruction on the test set (FAD and SI-SDR).
2. Re-introduce mixing supervision as an **addition** on top of the clean v1 recipe,
   not as a replacement for it.
3. Evaluate whether mixing-equivariance can be learned without degrading the FAD ceiling
   the v1 baseline establishes.
