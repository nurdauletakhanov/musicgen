# Latent Arithmetic Measures the Origin

Code and evaluation data for **"Latent Arithmetic Measures the Origin: Recentering Audio Autoencoders Without Retraining"** (submitted to ICASSP 2027).

**The one-line result.** Stem removal by latent subtraction,
`g(f(mix) − f(stem))`, is measured in coordinates the model never fixes.
Shifting the latent origin by the encoding of silence, `f(0)`, leaves
reconstruction and every unit-sum latent combination *exactly* unchanged for
any encoder and decoder, yet it moves subtraction scores by decibels: it
removes 87–95 % of the apparent subtraction advantage of mixing supervision on
a pre-registered held-out corpus, and it improves stem subtraction on models we
never trained (DAC, EnCodec, Music2Latent, the public Lin-CAE baselines) with
no retraining. What mixing supervision genuinely buys is *convex* mixing, which
the shift provably cannot touch. **Recentre before you subtract, whatever the
model.**

The repo also trains the waveform GAN autoencoders used in the paper, with an
explicit **decode-mixing loss** that improves convex mixing by +1.8 dB at no
reconstruction cost (replicated across training seeds).

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
| **GAN AE, +ℒ_dec+disc (v3.1)** | 15.3× | +9.9 | **+8.6** | **1.046** | 0.061 |

Downstream — **stem removal by latent subtraction** (MUSDB18 test, SI-SDR dB):

| Model | drums | bass | vocals | other | all |
|---|---|---|---|---|---|
| 7.66× no mix | +4.2 | +0.4 | +5.2 | +3.9 | +3.4 |
| 7.66× +ℒ_dec | +5.7 | +4.9 | +7.2 | +5.1 | **+5.7** |
| 15.3× no mix | +1.2 | −1.6 | +3.0 | +2.1 | +1.2 |
| 15.3× +ℒ_dec+disc | +5.4 | +4.7 | +6.9 | +3.9 | **+5.2** |

**Update (Sep 2026, after review):** the subtraction advantage above is a
*latent-offset* effect, not a property of the loss. Subtraction uses
coefficients (1, −1), which mixing-equivariance (coefficient sum 1) does not
cover, so an affine offset in the latent map survives it. Adding the encoded
silence f(0) restores it. For an **affine** encoder f(x) = Ax + b the identity
is exact:

```
f(mix) − f(stem) + f(0) = f(res)
```

so the corrected decode is the model's own reconstruction of the residual, for
any decoder; matching the residual waveform additionally needs exact
reconstruction. Empirically this lifts **every evaluated GAN configuration** to
within 0.4–0.5 dB of its ceiling in aggregate (≈ +8.3 dB at 7.66×, ≈ +7.3 dB at
15.3×; per track the tail is wider — worst 2.2 dB, 29–32 of 49 tracks within
0.5 dB), and it removes most of the mixing-loss advantage on subtraction rather than
all of it. On this 49-track test set the residual is −0.03 dB (95%
track-cluster CI [−0.15, +0.10]), indistinguishable from zero; on a
pre-registered held-out corpus of 240 MoisesDB recordings, where the intervals
are about three times tighter, correction removes 87–95% of the advantage and
leaves a small but resolvable residue of +0.29, +0.12 and +0.42 dB, with
intervals excluding zero. See `evaluation/holdout_moisesdb/`.

**So: the loss's demonstrated effect is on _convex_ mixing (the tables above).
Stem subtraction is governed by the latent origin, not by the loss.** Reproduce
with `python -m evaluation.compute_subtraction --origin-correct`.

**Listen:** [audio examples](https://nurdauletakhanov.github.io/musicgen/) —
stem removal on MUSDB18 test mixtures, all four models side by side.
Example selection is deterministic and independent of model outputs
(no cherry-picking; protocol on the page).

Two findings worth flagging for anyone building on this:

- **The decode-mixing loss is the active ingredient** for convex mixing. Adding
  a discriminator on the mixed path inflates the decode-vs-decode metric
  (12.1 → 16.0 dB) while giving *no* ground-truth gain. (Whether the
  discriminator does anything on its own, without the loss, is untested.)
- **The commonly used decode-vs-decode SI-SDR_lin is confounded** by decoder
  phase variance. On a consistency-model decoder the same checkpoint scores
  ≈ −8 dB or *positive* depending purely on whether decode noise is shared.
  Report the ground-truth-referenced variant. Decoding the *same* latent twice —
  no latent arithmetic at all — already scores −1.8 dB under independent noise,
  so the decoder's own sampling accounts for most of the collapse. See
  [`scripts/_diag_old_vs_new_eval.py`](scripts/_diag_old_vs_new_eval.py) and
  [`scripts/_diag_same_latent.py`](scripts/_diag_same_latent.py).

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
| `v3.1` | from-scratch + ℒ_dec + disc-on-mix | scratch | 64 | 15.3× | 250k |

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

# 2. Train. v2.x are 25k-step fine-tunes of the converged v1.1 baseline, so
#    they need --warm-start; without it you train a different experiment from
#    random initialisation.
python -m training.train --config configs/experiments/v2/v2.1_decmix.yaml \
    --warm-start checkpoints/v1.1/best.pth

#    For NEW training rather than reproducing the paper, carve a real
#    validation split first (the released runs validated on the test split):
#    python -m scripts.make_val_split && set data.val_split: val in the config

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
- In **our Music2Latent fine-tuning experiment**, latent mixing error
  decreases and ground-truth mixing SI-SDR improves (−9.5 → −7.8 dB against a
  continued-training control), while waveform agreement remains poor. The two
  systems differ in encoder, loss family, compression and training data as
  well as decoder class, so this is a limit we measured on that experiment,
  not a demonstrated property of consistency autoencoders in general.

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
  title     = {Latent Arithmetic Measures the Origin: Recentering Audio Autoencoders Without Retraining},
  author    = {Akhanov, Nurdaulet},
  booktitle = {Proc. IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

## Recentering models we did not train

`evaluation/codec_run_subtraction.py` applies the same raw / `+f(0)` stem
subtraction to third-party autoencoders through small adapters
(`evaluation/codec_adapters.py`): DAC-44k and EnCodec-24k from `transformers`
(continuous encoder output, quantizer bypassed on every path), and the public
Lin-CAE family of Torres et al. (`pip install linear-cae`; ids `lin-cae`,
`lin-cae-2`, and their retrained `m2l`). Units are identical to the GAN runs,
so `evaluation.paired_stats` / `origin_effect` apply unchanged
(`--key dd` gives the phase-cancelled reading that consistency decoders need).

```bash
python -m evaluation.codec_run_subtraction --model dac44k --out evaluation/v2_metrics/codecs/dac44k_subtraction.json \
    --per-chunk-out evaluation/v2_metrics/codecs/per_chunk/dac44k_per_chunk.json
python -m evaluation.codec_run_subtraction --model dac44k --origin-correct ...   # +f(0)
sbatch scripts/slurm_codec_origin.sh                                             # everything, both corpora
```

MUSDB18 is DAC training data and MoisesDB is Lin-CAE training data, so each
model is reported on the corpus it has not seen (`evaluation/v2_metrics/codecs/`
for MUSDB18 test, `evaluation/holdout_moisesdb/codecs/` for MoisesDB). Results
table: paper Table III (`research/paper/icassp2027/tables/codecs.tex`).

## Held-out evaluation (MoisesDB)

`evaluation/holdout_moisesdb/` holds the pre-registered held-out run in full:
per-unit records for every model in both raw and recentered form
(`per_chunk/*_per_chunk.json`, 28,800 units each), the per-unit convex-mixing
records naming both recordings of each pair, the target-activity measurements
that implement the silent-target rule (`activity.json`), the corpus manifest
(`corpus_manifest.json`, 240 tracks with per-track length and any empty stem
group), and every statistic computed from them. The protocol was committed
before the audio was downloaded: `research/paper/icassp2027/HOLDOUT_PROTOCOL.md`.

Regenerate the statistics from the records alone:

```bash
python -m evaluation.origin_effect \
    --per-chunk 'evaluation/holdout_moisesdb/per_chunk/*_per_chunk.json' \
    --activity evaluation/holdout_moisesdb/activity.json \
    --out /tmp/origin_effect.json
python -m evaluation.mixing_stats \
    --per-unit 'evaluation/holdout_moisesdb/per_chunk/*_mixing_per_unit.json' \
    --out /tmp/mixing_stats.json
```

Rebuild the corpus from the MoisesDB download (`scripts/prepare_moisesdb.py`)
to reproduce the records themselves.
