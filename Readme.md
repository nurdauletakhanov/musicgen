# Mixing-Equivariant Audio Autoencoders

Code and evaluation data for **"An Explicit Decode-Mixing Loss for
Mixing-Equivariant Audio Autoencoders"** (submitted to ICASSP 2027).

Audio mixing is linear in the waveform domain, but neural autoencoder latents do
not preserve that structure: interpolating two latent codes and decoding does
not reproduce the corresponding waveform mix. This repo trains waveform
autoencoders with an explicit **decode-mixing loss** that enforces

```
g( α·f(x₁) + (1−α)·f(x₂) )  ≈  α·x₁ + (1−α)·x₂
```

directly, at the cost of one extra decoder pass per step. The result is a latent
space where vector arithmetic is audio editing — subtracting a stem's latent
removes it from a mixture — **at no measured reconstruction cost**.

Paper source (LaTeX, tables, figures): [`research/paper/icassp2027/`](research/paper/icassp2027/).
Full experiment log and per-source breakdowns: [`evaluation/v2_metrics/RESULTS.md`](evaluation/v2_metrics/RESULTS.md).
**To reproduce a specific number, table, or figure from the paper, see
[`REPRODUCING.md`](REPRODUCING.md).**

---

## Headline results

Full multi-source test set, α=0.5, single seed. SDR$_{lin}^{gt}$ is
SI-SDR(g(z̄), x̄) — equivariance against the **ground-truth** mix.

| Model | Compression | SDR_rec ↑ | SDR_lin_gt ↑ | MixRate ↓ | FAD ↓ |
|---|---|---|---|---|---|
| GAN AE, no mix (v2.0) | 7.66× | +11.3 | +8.0 | 1.203 | 0.045 |
| **GAN AE, +ℒ_dec (v2.1)** | 7.66× | +11.4 | **+10.2** | **0.972** | 0.044 |
| GAN AE, no mix (v3.0) | 15.3× | +10.0 | +6.8 | 1.187 | 0.052 |
| **GAN AE, +mix (v3.1)** | 15.3× | +9.9 | **+8.6** | **1.046** | 0.061 |

Downstream — **stem removal by latent subtraction** (MUSDB18 test, SI-SDR dB):

| Model | drums | bass | vocals | other | all |
|---|---|---|---|---|---|
| 7.66× no mix | +4.2 | +0.4 | +5.2 | +3.9 | +3.4 |
| 7.66× +ℒ_dec | +5.7 | +4.9 | +7.2 | +5.1 | **+5.7** |
| 15.3× no mix | +1.2 | −1.6 | +3.0 | +2.1 | +1.2 |
| 15.3× +mix | +5.4 | +4.7 | +6.9 | +3.9 | **+5.2** |

At 15.3× compression the no-mixing control is *unusable* for latent subtraction
(−1.6 dB on bass — worse than passing the mixture through untouched). The loss
is what makes the operation viable, not merely better.

Two findings worth flagging for anyone building on this:

- **The decode-mixing loss is the active ingredient.** Adding a discriminator on
  the mixed path inflates the decode-vs-decode metric (12.1 → 16.0 dB) while
  giving *no* ground-truth gain. No adversarial machinery is required.
- **The commonly used decode-vs-decode SI-SDR_lin is confounded** by decoder
  phase variance. On a consistency-model decoder the same checkpoint scores
  ≈ −8 dB or *positive* depending purely on whether decode noise is shared.
  Report the ground-truth-referenced variant. See
  [`scripts/_diag_old_vs_new_eval.py`](scripts/_diag_old_vs_new_eval.py).

---

## Model lineage

| Run | What it is | Init | d_model | Compression | Steps |
|---|---|---|---|---|---|
| `v1.1` | recon-only baseline | scratch | 128 | 7.66× | 250k |
| `v2.0` | continued-training control (no mixing) | v1.1 | 128 | 7.66× | +25k |
| `v2.1` | **ℒ_dec only** — the paper's recommended recipe | v1.1 | 128 | 7.66× | +25k |
| `v2.2` | ℒ_dec + discriminator-on-mix | v1.1 | 128 | 7.66× | +25k |
| `v2.3–2.5` | ℒ_enc only (γ = 5 / 10 / 20) | v1.1 | 128 | 7.66× | +25k |
| `v2.6` | ℒ_dec, frozen encoder | v1.1 | 128 | 7.66× | +25k |
| `v3.0` | matched from-scratch control (no mixing) | scratch | 64 | 15.3× | 250k |
| `v3.1` | from-scratch + mixing | scratch | 64 | 15.3× | 250k |

Configs for every run are tracked under [`configs/experiments/`](configs/experiments/).

### Architecture

33.5M parameters, wave-to-wave, no transformer, 44.1 kHz on 1-second mono chunks.

```
waveform [B, 1, 44100]
  encoder (DAC-style)   -> latent [B, 45, d_model]
    4 stages, strides [4, 5, 7, 7] = 980
    dilated resblocks (1, 3, 9) per stage, weight-norm Conv1d
  decoder (HiFi-GAN V1) -> waveform [B, 1, 44100]
    ConvTranspose1d upsampling + MRF blocks (k=3,7, dilations [1,3])
```

Discriminator is MSSTFTD (multi-scale STFT, three FFT sizes) + MPD (periods
2, 3, 5, 7, 11), enabled after `disc_start_step = 10,000`.

**Losses.** Multi-resolution STFT (1.0) + mel (1.0) + latent L2 (0.001) +
adversarial (0.5) + feature matching (2.0), plus the mixing terms:

- **ℒ_dec** (decode-mixing) — `L_recon(g(z̄), x̄)` for a latent interpolation z̄
  and the true waveform mix x̄. Gradients flow through the decoder *and* back
  into the encoder via z̄. One extra decoder pass per step.
- **ℒ_enc** (encoder-only) — `‖f(x̄) − z̄‖²`. One extra encoder pass, no decoder
  pass. Linearizes the encoder but leaves the decoder non-equivariant; reported
  as an ablation, not recommended.

---

## Data

Three sources in one unified chunk directory with a shared `index.json`. Every
entry carries a `source` tag (`fma` / `maestro` / `musdb`), which is what lets
evaluation report per-domain metrics without re-running on separate directories.

Chunks are 1 s @ 44.1 kHz (44,100 samples), stored `fp16` with the per-chunk
peak retained, peak-normalized to 0.95, non-overlapping on a fixed grid.

| Split | Source | Tracks | Chunks | Hours |
|---|---|---|---|---|
| train | FMA-large (train+val) | 95,065 | 2,767,005 | 768.6 |
| train | MAESTRO v3 (train+val) | 1,099 | 633,858 | 176.1 |
| train | MUSDB18-HQ train | 100 | 22,096 | 6.1 |
| test | FMA-large (official test) | 11,239 | 325,954 | 90.5 |
| test | MAESTRO test | 177 | 71,214 | 19.8 |
| test | MUSDB18-HQ test | 50 | 12,055 | 3.3 |

**950.8 h train / 113.7 h test.** MUSDB stems stay on disk — v1 uses only
`mixture.wav`, but the stem-subtraction eval needs the individual stems.

Sources: [FMA-large](https://github.com/mdeff/fma) (93 GB MP3, official 80/10/10
split via `fma_metadata/tracks.csv`; 270 tracks skipped for known-bad data),
[MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro),
[MUSDB18-HQ](https://zenodo.org/record/3338373). None are redistributed here.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Preprocess (resumable; --force to regenerate)
python -m data.preprocess musdb
python -m data.preprocess maestro
python -m data.preprocess fma --workers 8
python -m scripts.reshuffle_fma_splits    # move FMA's official test split into test/

# 2. Train
python -m training.train --config configs/experiments/v2/v2.1_decmix.yaml

# 3. Resume
python -m training.train --config configs/experiments/v2/v2.1_decmix.yaml \
    --resume ./checkpoints/v2.1-decmix/latest.pth
```

Outputs land in `checkpoints/<run>/`: `config.yaml` (copy of the run's config),
`train_log.jsonl`, `val_log.jsonl`, `best.pth`, `latest.pth`, and listenable
`samples/step_<N>/*.wav` reference/reconstruction pairs at each val checkpoint.

## Pretrained weights

Inference checkpoints (`best.pth` per run) are on the Hugging Face Hub:
**[`SoMa25/mixing-equivariant-ae-checkpoints`](https://huggingface.co/SoMa25/mixing-equivariant-ae-checkpoints)**.
Download them into `checkpoints/<run>/` and the eval drivers in
[`REPRODUCING.md`](REPRODUCING.md) will pick them up.

---

## Scope and limitations

Stated plainly, so nobody rediscovers these the hard way:

- **Single seed.** Every number is n=1. Differences under ~0.5 dB (e.g. v2.1 vs
  v2.2) are within plausible seed variance and are reported as "comparable."
- **Mono, 1-second chunks.** No stereo, no long-context modeling.
- **No listening test.** FAD is the only perceptual proxy.
- **The Music2Latent cross-architecture experiments (paper Section IV-B) are
  not reproducible from this repo.** That fine-tuning code lives in a separate
  `music2latent-mix` repo which is not public. What *is* here: the evaluation
  adapter ([`evaluation/m2l_adapter.py`](evaluation/m2l_adapter.py)), the eval
  drivers, and every resulting metric JSON — so the numbers are inspectable and
  the eval protocol is auditable, but the fine-tuning runs cannot be repeated.
- Latent arithmetic on a **consistency decoder stays phase-limited**; the
  geometry improves but waveform arithmetic does not become clean. This is a
  property of the decoder class, not of the loss.

## Layout

```
data/          dataset.py, preprocess.py (musdb | maestro | fma)
models/        encoder.py, decoder.py, autoencoder.py, discriminator.py
training/      trainer.py (step-based), train.py, config.py
configs/       base.yaml + experiments/{v1,v2,v3}/*.yaml
evaluation/    compute_{mixing_metrics,fad,subtraction}.py, m2l_* adapters
               v2_metrics/    all result JSONs + RESULTS.md
scripts/       run_{subtraction,alpha_sweep,v2_fad,v2_mixing_metrics}.py drivers
research/      paper/icassp2027/ — LaTeX source, table/figure generators
```

Scripts prefixed `_` are diagnostics rather than pipeline steps. One of them is
load-bearing: `_diag_old_vs_new_eval.py` produces the paper's Fig. 1.

## Citation

```bibtex
@inproceedings{akhanov2027mixing,
  title     = {An Explicit Decode-Mixing Loss for Mixing-Equivariant Audio Autoencoders},
  author    = {Akhanov, Nurdaulet},
  booktitle = {Proc. IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```
