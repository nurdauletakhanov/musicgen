# v2 Experiments — Results Inventory

Source: `evaluation/v2_metrics/*.json`. All checkpoints at step 25000, all evals at α=0.5, n=1 seed each.

Eval set sizes: FMA 26859 / Maestro 7044 / MUSDB 1181 / all 35084 embeddings for FAD; 409216 samples for mixing metrics.

## Metric definitions

From `evaluation/compute_mixing_metrics.py` and `evaluation/compute_fad.py`:

- **sdr_rec** — standard reconstruction SI-SDR (encode → decode), dB. Higher is better.
- **sdr_lin** — latent-arithmetic SI-SDR, **decode-vs-decode**. Encode pair (x₁, x₂), form z̄ = α·z₁ + (1−α)·z₂, then SI-SDR(g(z̄), g(f(x̄))) where x̄ = α·x₁ + (1−α)·x₂ — i.e. the decoded interpolation is compared to the model's *own reconstruction of the true mix*, NOT to x̄ itself (`compute_mixing_metrics.py` `_process_batch`). dB, higher is better. Measures mixing-equivariance relative to the model's reconstruction ceiling; it is NOT comparable to papers that report SI-SDR vs the ground-truth mix (Torres et al., Music2Latent Table 3). Caveat: a decoder that grows less sensitive to its latent inflates this metric (with shared decode noise, a latent-ignoring decoder scores +∞); cross-check against `mix_rate` and `sdr_lin_gt`.
- **sdr_lin_gt** — latent-arithmetic SI-SDR vs **ground truth**: SI-SDR(g(z̄), x̄). The externally comparable variant (added June 2026; older JSONs lack it until re-run).
- **l_lat** — MSE between E(x̄) and z̄ in latent space. Lower is better. (Encoder linearity in latent space.)
- **mix_rate** — ratio of recon loss on mixed-decoded path vs encode-then-decode-of-mix path. <1 means latent mix beats encoded mix.
- **fad** — Fréchet Audio Distance vs ground truth, LAION-CLAP backbone, 10 s clips at 48 kHz. Lower is better.

## Headline table (all-source aggregate)

Sorted by run number, baseline first.

| # | Run | Loss config | Frozen enc? | FAD ↓ | sdr_rec ↑ | sdr_lin ↑ | Δ sdr_lin vs v2.0 | mix_rate ↓ | l_lat ↓ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | v2.0-continued | none (control) | — | 0.04505 | 11.33 | 11.996 | — | 1.203 | 0.0678 |
| 1 | v2.1-decmix | ℒ_dec (w=0.5) | No | 0.04425 | 11.35 | 12.128 | +0.13 | 0.972 | 0.0520 |
| 2a | v2.2-decmix-disc-old-asym | ℒ_dec + disc-on-mix (asym) | No | 0.04432 | 11.29 | **15.736** | **+3.74** | 1.060 | 0.0505 |
| 2b | v2.2-decmix-disc (sym) | ℒ_dec + disc-on-mix (sym) | No | **0.04398** | 11.38 | **15.866** | **+3.87** | 1.054 | 0.0493 |
| 3 | v2.3-encmix-g5 | ℒ_enc (γ=5) | No | 0.04372 | 11.32 | 12.817 | +0.82 | 1.175 | 0.0441 |
| 4 | v2.4-encmix-g10 | ℒ_enc (γ=10) | No | 0.04421 | 11.30 | 13.129 | +1.13 | 1.165 | 0.0393 |
| 5 | v2.5-encmix-g20 | ℒ_enc (γ=20) | No | 0.04527 | 11.30 | 13.506 | +1.51 | 1.153 | 0.0340 |
| 6 | v2.6-decmix-frozenenc | ℒ_dec (w=0.5), frozen enc | **Yes** | 0.04572 | 11.21 | 11.718 | **−0.28** | 1.056 | 0.0720 |

(Bold = headline numbers. All sdr_rec, sdr_lin in dB.)

## What the numbers say (factual observations only)

- **sdr_lin spans 11.7 → 15.9 dB** across the 8 runs — a 4.2 dB range.
- **sdr_rec spans 11.21 → 11.38 dB** — essentially flat. Reconstruction quality is preserved across all loss configurations.
- **FAD spans 0.0437 → 0.0457** — ~5% variation. Not discriminative at this scale, but v2.2 sym (0.04398) is third-lowest and beats baseline (0.04505).
- **v2.2 (decmix + disc-on-mix) is the largest jump on sdr_lin**: +3.74 dB (asym), +3.87 dB (sym).
- **v2.2 sym vs asym differ by 0.13 dB on sdr_lin**, 0.0012 on l_lat, 0.0003 on FAD (sym slightly better on all three).
- **v2.6 (frozen encoder + ℒ_dec) is the only run with sdr_lin below v2.0** (−0.28 dB).
- **ℒ_enc gain monotonic in γ on sdr_lin**: 12.82 → 13.13 → 13.51 dB for γ ∈ {5, 10, 20}.
- **l_lat decreases monotonically with γ**: 0.0441 → 0.0393 → 0.0340 (v2.3 → v2.4 → v2.5).
- **Only v2.1 has mix_rate < 1** (0.972). All other runs have mix_rate in [1.05, 1.20].
- **v2.1 has the smallest sdr_lin gain (+0.13)** despite the lowest mix_rate. mix_rate and sdr_lin are not co-monotonic across runs.
- **v2.6 (frozen) has worse l_lat than v2.0** (0.0720 vs 0.0678) despite being trained with ℒ_dec.

## Corrected-metric re-eval (June 2026) — sdr_lin_gt changes the mechanism reading

The headline above (v2.2 disc-on-mix +3.87 dB) was computed on **sdr_lin = SI-SDR(g(z̄), g(f(x̄)))** — decode-vs-decode, which measures consistency between two latent routes, NOT the mixing-equivariance equation g(z̄) ≈ x̄. Re-running all 7 v2 models with **sdr_lin_gt = SI-SDR(g(z̄), x̄)** (vs the true mix — the metric that directly measures the claimed property; deterministic GAN decoder so it's phase-coherent and clean):

| model | sdr_lin (dd, old) | **sdr_lin_gt (vs GT)** | Δ_gt vs v2.0 | mix_rate ↓ |
|---|---|---|---|---|
| v2.0 baseline | 12.00 | 7.99 | — | 1.203 |
| v2.1 decmix only | 12.13 | **10.25** | **+2.26** | **0.972** |
| v2.2 decmix + disc | **16.03** | 9.89 | +1.90 | 1.052 |
| v2.3 encmix g5 | 12.82 | 8.37 | +0.38 | 1.175 |
| v2.4 encmix g10 | 13.13 | 8.50 | +0.51 | 1.165 |
| v2.5 encmix g20 | 13.51 | 8.69 | +0.70 | 1.153 |
| v2.6 frozen-enc | 11.72 | 9.54 | +1.55 | 1.056 |

**Observations (factual):**
- Two independent ground-truth-based metrics agree: **sdr_lin_gt** and **mix_rate** both rank v2.1 (decmix only) at or above v2.2 (decmix + disc). The disc-on-mix's +3.87 dB advantage exists ONLY in the decode-vs-decode metric.
- The decode-vs-decode → ground-truth gap widens with disc-on-mix: v2.0 gap = 4.0 dB, v2.2 gap = 6.1 dB. Interpretation: disc-on-mix pulls g(z̄) and g(f(x̄)) onto a common realistic manifold (raising their mutual SI-SDR) without bringing g(z̄) closer to the true mix x̄.
- v2.1 vs v2.2 on sdr_lin_gt is 10.25 vs 9.89 = **0.36 dB, within plausible n=1 seed noise** → defensible statement is "disc-on-mix does not measurably improve ground-truth equivariance," NOT "v2.1 is best."

**Framing (CONFIRMED — v2.1 subtraction in):** headline = decode-mixing loss drives equivariance (every mixing model beats baseline on sdr_lin_gt). Disc-on-mix is an ablation that does NOT add a consistent ground-truth benefit → the plain loss is sufficient, no adversarial machinery required (consistent with M2L, which used no discriminator).

Triangulated across three ground-truth metrics, v2.1 (decmix only) ≈ v2.2 (decmix + disc), no consistent winner:

| model | sdr_lin_gt | mix_rate ↓ | subtraction (vs GT) | gap ↓ | sdr_dd (dd) |
|---|---|---|---|---|---|
| v2.0 baseline | 7.99 | 1.203 | +3.42 | +5.21 | +6.42 |
| v2.1 decmix only | **10.25** | **0.972** | +5.73 | +2.97 | +10.01 |
| v2.2 decmix + disc | 9.89 | 1.052 | **+6.04** | **+2.59** | +11.77 |

v2.1 leads on sdr_lin_gt + mix_rate; v2.2 leads on subtraction by 0.31 dB (within n=1 noise) and sdr_dd. Both ~halve the linearity-tax gap vs baseline. Disc-on-mix's only consistent advantage is on sdr_dd (decode-vs-decode, phase-cancelled — same family as the inflated sdr_lin), which is not a ground-truth equivariance measure. **Conclusion: the decode-mixing loss is the active ingredient; the discriminator-on-mix is unnecessary for mixing-equivariance.**

## Per-source breakdown

### v2.0-continued (control, no mixing loss)
| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.0421 | 0.0844 | 0.0568 | 0.04505 |
| sdr_rec | 9.63 | 19.62 | 8.58 | 11.33 |
| sdr_lin | 11.16 | 16.04 | 10.67 | 12.00 |
| mix_rate | 1.203 | 1.210 | 1.183 | 1.203 |
| l_lat | 0.0712 | 0.0520 | 0.0676 | 0.0678 |

### v2.1-decmix (ℒ_dec only, w=0.5)
| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.0416 | 0.0828 | 0.0538 | 0.04425 |
| sdr_rec | 9.64 | 19.65 | 8.63 | 11.35 |
| sdr_lin | 10.60 | 19.50 | 9.87 | 12.13 |
| mix_rate | 0.962 | 1.024 | 0.927 | 0.972 |
| l_lat | 0.0546 | 0.0398 | 0.0531 | 0.0520 |

### v2.2-decmix-disc-old-asym (ℒ_dec + disc on mix, asymmetric disc)
| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.0426 | 0.0756 | 0.0658 | 0.04432 |
| sdr_rec | 9.60 | 19.46 | 8.56 | 11.29 |
| sdr_lin | 14.47 | 21.87 | 13.71 | 15.74 |
| mix_rate | 1.060 | 1.066 | 1.047 | 1.060 |
| l_lat | 0.0534 | 0.0368 | 0.0533 | 0.0505 |

### v2.2-decmix-disc (ℒ_dec + disc on mix, symmetric)
| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.0410 | 0.0816 | 0.0609 | 0.04398 |
| sdr_rec | 9.69 | 19.56 | 8.65 | 11.38 |
| sdr_lin | 14.57 | 22.12 | 13.91 | 15.87 |
| mix_rate | 1.054 | 1.057 | 1.041 | 1.054 |
| l_lat | 0.0521 | 0.0361 | 0.0521 | 0.0493 |

### v2.3-encmix-g5 (ℒ_enc only, γ=5)
| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.0402 | 0.0855 | 0.0550 | 0.04372 |
| sdr_rec | 9.62 | 19.56 | 8.61 | 11.32 |
| sdr_lin | 11.95 | 17.02 | 11.44 | 12.82 |
| mix_rate | 1.173 | 1.188 | 1.158 | 1.175 |
| l_lat | 0.0460 | 0.0351 | 0.0458 | 0.0441 |

### v2.4-encmix-g10 (ℒ_enc only, γ=10)
| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.0409 | 0.0849 | 0.0632 | 0.04421 |
| sdr_rec | 9.60 | 19.55 | 8.58 | 11.30 |
| sdr_lin | 12.24 | 17.43 | 11.73 | 13.13 |
| mix_rate | 1.163 | 1.178 | 1.148 | 1.165 |
| l_lat | 0.0410 | 0.0311 | 0.0409 | 0.0393 |

### v2.5-encmix-g20 (ℒ_enc only, γ=20)
| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.0421 | 0.0850 | 0.0614 | 0.04527 |
| sdr_rec | 9.61 | 19.52 | 8.60 | 11.30 |
| sdr_lin | 12.58 | 17.98 | 12.06 | 13.51 |
| mix_rate | 1.151 | 1.162 | 1.135 | 1.153 |
| l_lat | 0.0355 | 0.0266 | 0.0354 | 0.0340 |

### v2.6-decmix-frozenenc (ℒ_dec, frozen encoder)
| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.0413 | 0.0943 | 0.0528 | 0.04572 |
| sdr_rec | 9.51 | 19.44 | 8.53 | 11.21 |
| sdr_lin | 10.48 | 17.70 | 9.89 | 11.72 |
| mix_rate | 1.050 | 1.093 | 1.010 | 1.056 |
| l_lat | 0.0754 | 0.0565 | 0.0720 | 0.0720 |

## Cross-source patterns

- **Maestro is consistently the easiest source** across all runs (sdr_rec ~19.5 dB, sdr_lin 16-22 dB). Solo piano with strong harmonic structure.
- **MUSDB is consistently the hardest** (sdr_rec ~8.6 dB, sdr_lin 9.9-13.9 dB). Multi-track mixed music.
- **FMA sits in between** (sdr_rec ~9.6 dB, sdr_lin 10.5-14.6 dB).
- **The +3.7 dB v2.2 win on `all` shows up consistently per source**: FMA +3.3 to +3.4, Maestro +5.8 to +6.1, MUSDB +3.0 to +3.2. The disc-on-mix improvement is not driven by one source.

## Missing pieces in the data

- **No multi-seed.** All numbers are n=1.
- **No phase error.** Metric not implemented in the codebase.
- **α=0.5 only.** No sweep over α.
- **No standard-deviation bars** anywhere — single eval per run.
- ~~**No M2L or other-architecture baseline.**~~ — see M2L generalization section below.

---

# M2L generalization — Phase 0/1/2 fine-tuning

Fine-tuning the published Music2Latent (Pasini 2024) checkpoint with the v2 mixing-linearity losses, to test whether the v2 finding transfers to a structurally different autoencoder. M2L is a 2D STFT encoder/decoder + consistency-model UNet (vs v2's 1D waveform GAN encoder/decoder). All M2L runs evaluate on the same test set as the v2 rows (FMA/MAESTRO/MUSDB chunks-44k-1s, 409,216 samples for mixing metrics, 35,084 CLAP embeddings for FAD).

## Setup

- **Phase 0**: vanilla published M2L checkpoint, eval-only. M2L row 0 of the headline table.
- **Phase 1**: fine-tune from M2L checkpoint with **ℒ_enc** (encoder mixing-linearity loss, γ=5000). Mirrors v2.5. Trained 25,000 optimizer steps at batch=16, lr=1e-5.
- **Phase 2**: fine-tune from M2L checkpoint with **ℒ_dec + consistency-on-mix**. Mirrors v2.2-sym in spirit, but the v2 disc-on-mix discriminator is replaced by an M2L-native analog: applying M2L's pseudo-Huber consistency loss to the mixed-pair path (`x̄`, `z_interp`) so mixed latents enter the decoder's training distribution. Trained 25,000 optimizer steps (50,000 micro-batches × accum=2) at batch=8 effective-batch=16, lr=1e-5.

Code lives in [`d:/projects/music2latent-mix`](d:/projects/music2latent-mix) on the `mix-linearity` branch off `origin/training`. Eval scripts in [`evaluation/m2l_run_mixing.py`](d:/projects/musicgen/evaluation/m2l_run_mixing.py) / [`m2l_run_fad.py`](d:/projects/musicgen/evaluation/m2l_run_fad.py) wrap an [`M2LAutoencoderAdapter`](d:/projects/musicgen/evaluation/m2l_adapter.py) that exposes M2L behind the v2 `Autoencoder` interface (so the existing `compute_mixing_metrics` and `compute_fad` work unchanged).

## Headline table — M2L Phase 0/1/2/2b (deterministic-noise eval)

| # | Run | Loss config | FAD ↓ | sdr_rec ↑ | sdr_lin ↑ | mix_rate ↓ | l_lat ↓ |
|---|---|---|---|---|---|---|---|
| M0 | M2L Phase 0 | none (baseline) | 0.1269 | −3.28 | 5.02 | 1.101 | 0.150 |
| M1 | M2L Phase 1 | ℒ_enc (γ=5000) | 0.1360 | −3.14 | 5.82 | 1.120 | **0.106** |
| M2 | M2L Phase 2 | ℒ_dec (w=0.5) + cons-on-mix (w=1.0) | **0.1190** | −2.60 | **6.91** | 1.090 | 0.115 |
| M2b | M2L Phase 2b | ℒ_dec (**w=5.0**) + cons-on-mix (w=1.0) | 0.2840 | **−0.52** | 5.68 | **1.067** | 0.111 |

(Bold = best across M2L runs. All sdr in dB.)

**Phase 2b is NOT a strict improvement over Phase 2** — sdr_lin and FAD both regressed when `decode_mix_weight` was bumped 10× (0.5 → 5.0). See the "ℒ_dec weight Pareto frontier" section below.

### Per-source breakdown

**Phase 0 (M2L baseline)** — [m2l_phase0_mixing.json](m2l_phase0_mixing.json) / [m2l_phase0_fad.json](m2l_phase0_fad.json)

| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.1252 | 0.1662 | 0.1888 | 0.1269 |
| sdr_rec | −4.25 | +1.15 | −3.39 | −3.28 |
| sdr_lin | 5.01 | 5.23 | 4.14 | 5.02 |
| mix_rate | 1.109 | 1.063 | 1.091 | 1.101 |
| l_lat | 0.157 | 0.110 | 0.183 | 0.150 |

**Phase 1 (ℒ_enc γ=5000)** — [m2l_phase1_encmix_mixing.json](m2l_phase1_encmix_mixing.json) / [m2l_phase1_encmix_fad.json](m2l_phase1_encmix_fad.json)

| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.1259 | 0.2076 | 0.2059 | 0.1360 |
| sdr_rec | −4.09 | +1.30 | −3.47 | −3.14 |
| sdr_lin | 5.82 | 5.90 | 5.26 | 5.82 |
| mix_rate | 1.124 | 1.103 | 1.101 | 1.120 |
| l_lat | 0.111 | 0.079 | 0.113 | 0.106 |

**Phase 2 (ℒ_dec + cons-on-mix)** — [m2l_phase2_decmix_consmix_mixing.json](m2l_phase2_decmix_consmix_mixing.json) / [m2l_phase2_decmix_consmix_fad.json](m2l_phase2_decmix_consmix_fad.json)

| metric | fma | maestro | musdb | all |
|---|---|---|---|---|
| FAD | 0.1209 | 0.1436 | 0.1902 | 0.1190 |
| sdr_rec | −3.55 | +1.83 | −2.91 | −2.60 |
| sdr_lin | 6.83 | 7.39 | 6.31 | 6.91 |
| mix_rate | 1.091 | 1.089 | 1.057 | 1.090 |
| l_lat | 0.122 | 0.077 | 0.130 | 0.115 |

### Phase 2 deltas vs Phase 0

- sdr_lin: **+1.89 dB** all (+1.82 fma / +2.16 maestro / +2.17 musdb) — uniform across sources
- l_lat: **−23%** (0.150 → 0.115) — encoder more linear despite no ℒ_enc training (cons-on-mix backprops through `z_interp = α·f(x₁) + (1−α)·f(x₂)` and induces some encoder linearity as a side effect)
- mix_rate: **−1%** (1.101 → 1.090) — decoder mixing-equivariance improved
- FAD: **−6%** (0.127 → 0.119) — recon quality *improved*, not regressed
- sdr_rec: **+0.68 dB** (−3.28 → −2.60) — phase-incoherent baseline shifts modestly

**No quality trade-off**: Phase 2 strictly dominates Phase 0 on every mixing-equivariance metric **and** improves FAD.

### Phase 1 vs Phase 2 — why the encoder-only intervention isn't enough

Phase 1 (ℒ_enc only) showed a textbook trade-off:
- ✓ Encoder linearity ↑ (l_lat −29%, sdr_lin +0.80 dB)
- ✗ FAD regressed +7% (maestro alone +25%)
- ✗ mix_rate regressed +2%

Phase 1's encoder shifted its latent manifold to be more linear, but the decoder hadn't seen that distribution and decoded mixed latents slightly worse → audible quality drop on cleaner sources (maestro). Phase 2's cons-on-mix term fixed both regressions by training the decoder on the mixed-latent distribution.

**Take-away**: on architectures with a sensitive decoder (M2L's consistency model), encoder-only mixing pressure isn't enough — you also need a way to keep the decoder's training distribution covering mixed latents. v2.2-sym used a discriminator for this; on M2L we used the M2L-native consistency-on-mix substitute. **Same intent, different mechanism, same outcome.**

## Eval-protocol discrepancy with the old paper draft

The old paper draft (`research/paper/ismir/paper.tex`) reports M2L baseline `SI-SDR_lin = −8.7 dB` and "M2L + ℒ_dec (ours)" `= 11.7 dB` — a +20.4 dB jump. **Our M2L Phase 2 result (+1.89 dB above baseline) does NOT reproduce this number.** Diagnosed root cause:

### Two eval protocols, two answers

M2L's decoder is a **consistency model**: every `decode(z)` call starts from `randn(...) * sigma_max` and the random initial noise determines the **time-domain phase** of the output. The latent fixes content; the noise fixes phase.

`SI-SDR_lin = SI-SDR(g(z̄), g(f(x̄)))` compares two decode calls. Two protocols:

- **Random-noise protocol** (old paper draft, presumably also Torres et al.): each decode uses **independent** random noise. Even if z̄ ≈ f(x̄), the two outputs have different phases → SI-SDR is dominated by phase variance, not real linearity. **Reports `~−8 dB regardless of true mixing-linearity.`**
- **Shared-noise protocol** (current [m2l_adapter.py:88-112](d:/projects/musicgen/evaluation/m2l_adapter.py#L88-L112)): both decodes use the **same** seeded random noise. Phase nuisance cancels. **Reports real mixing-linearity.**

### Diagnostic verifying both protocols ([scripts/_diag_old_vs_new_eval.py](d:/projects/musicgen/scripts/_diag_old_vs_new_eval.py))

240 MUSDB chunks at α=0.5:

| Protocol | M2L Phase 0 | M2L Phase 2 | Δ |
|---|---|---|---|
| **shared-noise** (mine) | +3.54 | +5.96 | **+2.42 dB** |
| **random-noise** (old paper) | **−9.28** | −7.62 | **+1.66 dB** |

The random-noise Phase 0 number (−9.28 dB) reproduces the old paper's −8.7 dB within sample noise, **confirming that's how the old eval was computed**. Both protocols agree: the *real* L_dec gain on M2L is roughly **+1.6 to +2.4 dB**, NOT +20.4 dB.

The fingerprint that gave the diagnosis away: the old `results/baseline_evaluation.json` shows a **U-shape across α** — `SI-SDR_lin = −1.7, −6.2, −8.8, −4.9, +0.2` for α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}. For real mixing-linearity this metric should be approximately α-independent. The U is the signature of a metric dominated by phase variance: at α=0.5 the latents `z̄` and `f(x̄)` differ most → phase decisions diverge most → SI-SDR collapses. At α=0.1/0.9 one stem dominates → latents almost identical → phase decisions agree → SI-SDR ~0.

### Where the old paper's `+11.7 dB` came from — unresolved

We could not reproduce the +11.7 number from the current Phase 2 checkpoint under either protocol. Possible explanations (each testable):

1. **Older M2L checkpoint** was used in the paper draft. Worth checking if there's an earlier Pasini release.
2. **20 epochs vs ~2 epochs of training**. Paper draft says 20 epochs of fine-tuning; our Phase 2 ran 25 k optimizer steps ≈ 2 epochs at the chunk-dataset size. Long L_dec training might gradually train the decoder to suppress phase variance from the random init noise — making the random-noise SI-SDR_lin self-determinizing. If so the +20.4 dB jump is *real* but reflects phase-determinism, not mixing-linearity.
3. **Old eval script differs from its own stated formula**. The script that produced `results/baseline_evaluation.json` is no longer in git (the v15 archive's `baseline_eval.py` only computes MixRate, not SI-SDR_lin). Possibly the eval averaged over multiple noise realizations or used a deterministic decode path.

# Conclusions

## Findings the v2 paper can claim with confidence

1. **The v2 mixing-linearity finding generalizes to a structurally different autoencoder.** The qualitative pattern (ℒ_enc improves encoder linearity, ℒ_dec + a way-to-keep-mixed-latents-in-distribution improves end-to-end mixing equivariance, both losses are compatible with reconstruction quality) reproduces directionally on M2L. **(Phases 0 → 1 → 2 above.)**

2. **The disc-on-mix idea isn't architecture-specific.** v2.2-sym achieved its sdr_lin gain through a discriminator on the mixed-decode output. On M2L (a consistency model with no native discriminator) we replaced this with an M2L-native analog — the consistency loss applied to the mixed pair. **Same intent, same outcome.** This supports the broader paper claim that mixing-equivariance is a property of the *training signal*, not a specific adversarial setup.

3. **Encoder-only mixing pressure can hurt reconstruction quality on architectures with a distribution-sensitive decoder.** Phase 1's FAD regression on maestro (+25%) is an architecture-specific finding worth reporting: on consistency-model decoders, the encoder-only intervention pulls the latent distribution out from under the decoder. Phase 2's cons-on-mix fixes this. v2's GAN decoder didn't show this effect (v2.5's FAD was unchanged from v2.0).

4. **Magnitude of improvement is smaller on M2L than on v2** (+1.89 vs +3.87 dB sdr_lin at α=0.5, deterministic-noise eval). Two likely reasons:
   - M2L's tanh-bottlenecked latents have a fundamental ceiling on how linear the encoder can become.
   - We trained ~10× less than the old paper draft claims (2 epochs vs 20 epochs of fine-tuning).

## Things the v2 paper should NOT claim without further work

1. **The `−8.7 → 11.7 = +20.4 dB` headline.** The −8.7 baseline is mostly phase-noise variance, not mixing-linearity. The +11.7 number does not reproduce from the Phase 2 checkpoint with the eval scripts currently in git. Either:
   - Switch the headline to a deterministic-noise number (`+5.02 → +6.91 = +1.89 dB`), with disclosure that the comparison vs Torres et al. (−8.7 → +2.3) is under a different protocol.
   - **Or** find the original eval script + checkpoint that produced +11.7 and verify it reproduces, before publishing.

2. **The "5× over Torres et al." framing.** Both `−8.7` (M2L baseline) and Torres et al.'s `+2.3` are under the random-noise protocol — internally consistent for that comparison. But the gain numbers compounded together (+11 dB for Torres, claimed +20 for ours) include phase-noise-suppression as part of the gain. Worth disclosing in methods either way, especially since our Phase 2 random-noise gain is +1.66 dB, not +20.

## Reproducibility — where each number lives

| Phase | Code | Config | Eval script | Result JSON |
|---|---|---|---|---|
| 0 | published M2L | n/a | [m2l_run_mixing.py](d:/projects/musicgen/evaluation/m2l_run_mixing.py) / [m2l_run_fad.py](d:/projects/musicgen/evaluation/m2l_run_fad.py) | [m2l_phase0_*.json](.) |
| 1 | [music2latent-mix `mix-linearity`](d:/projects/music2latent-mix) | [mix_phase1_encmix.py](d:/projects/music2latent-mix/configs/mix_phase1_encmix.py) | same as Phase 0 | [m2l_phase1_*.json](.) |
| 2 | same branch | [mix_phase2_decmix_consmix.py](d:/projects/music2latent-mix/configs/mix_phase2_decmix_consmix.py) | same as Phase 0 | [m2l_phase2_*.json](.) |
| Diagnostic | [scripts/_diag_old_vs_new_eval.py](d:/projects/musicgen/scripts/_diag_old_vs_new_eval.py) | n/a | (self-contained) | [_diag_old_vs_new_eval.log](_diag_old_vs_new_eval.log) |

Phase 1 and Phase 2 share the same `mix-linearity` branch — Phase 2 code is purely additive on top of Phase 1, gated by config (`decode_mix_weight + cons_mix_weight > 0`). Re-running Phase 1's config on the current code produces the same result as before Phase 2 was added.

---

# Phase 3 re-eval (EMA-consistent + new metrics, June 2026) — SUPERSEDES the M2L numbers above

The original M2L rows above had two problems (see the project audit): (a) the fine-tune phases were evaluated with **raw** `gen_state_dict` weights while Phase 0 used the published **EMA-merged** weights, and (b) `sdr_lin` is decode-vs-decode (g(z̄) vs g(f(x̄))), not comparable to papers reporting SI-SDR vs the ground-truth mix. Phase 3 fixes both: every fine-tune phase re-run against its `*_ema.pt` checkpoint (`music2latent-mix/scripts/make_ema_checkpoint.py`), and `sdr_lin_gt` = SI-SDR(g(z̄), x̄) added. Plus the new **Phase 0.5 control** (consistency-only fine-tune, no mixing losses, same data/schedule). Files: `m2l_phase{0,05,1,2,2b}_ema_{mixing,fad,subtraction}.json`.

## M2L headline (EMA weights, α=0.5)

| Run | Loss config | FAD ↓ | sdr_rec | sdr_lin (dd) | **sdr_lin_gt** | l_lat ↓ | mix_rate ↓ |
|---|---|---|---|---|---|---|---|
| Phase 0 | published baseline | 0.1269 | −3.28 | 5.02 | **−9.63** | 0.150 | 1.101 |
| **Phase 0.5** | **continued-train control, no mix** | 0.1339 | −3.09 | 5.08 | **−9.50** | 0.145 | 1.125 |
| Phase 1 | ℒ_enc (γ=5000) | 0.1364 | −3.12 | 5.84 | **−9.09** | 0.105 | 1.118 |
| **Phase 2** | ℒ_dec (0.5) + cons-on-mix (1.0) | **0.1189** | −2.57 | **6.92** | **−7.80** | 0.115 | 1.086 |
| Phase 2b | ℒ_dec (5.0) + cons-on-mix | 0.2926 | **−0.45** | 5.66 | **−4.62** | 0.111 | **1.064** |

(Phase 0 FAD is the published-checkpoint value — already EMA, not re-run. sdr_lin_gt is negative for all M2L runs: the consistency decoder is phase-incoherent with raw waveforms, so g(z̄) never aligns in phase with x̄ regardless of latent linearity.)

## M2L subtraction (EMA weights), with new phase-cancelled `sdr_dd` column

| Run | sub (vs GT) | ceil | **sdr_dd (phase-cancelled)** | gap |
|---|---|---|---|---|
| Phase 0 | −13.81 | −2.74 | **−3.18** | 11.07 |
| Phase 2 | −11.18 | −2.79 | **+0.39** | 8.39 |

## Decision-gate verdict (Phase 4)

1. **Control passes — gain is mixing supervision, not domain adaptation.** Phase 0.5 moves sdr_lin +0.06, sdr_lin_gt +0.13 dB over Phase 0, and FAD *worsens* (0.127→0.134). Phase 2's +1.9 dB (both protocols) with *better* FAD is therefore attributable to the mixing losses. **The cross-architecture section stays.**

2. **Both protocols agree on the improvement → not a measurement artifact.** Phase 2 vs Phase 0: +1.90 dB sdr_lin (decode-vs-decode) AND +1.83 dB sdr_lin_gt (vs ground truth). The earlier worry that decode-vs-decode inflated the gain is falsified — same direction, same magnitude. Subtraction confirms independently: sdr_dd −3.18 → +0.39 (+3.6 dB, phase nuisance removed).

3. **Honest limitation (now stated explicitly).** Absolute sdr_lin_gt stays ≈ −8 dB and subtraction-vs-GT ≈ −11 dB because M2L's consistency decoder is phase-incoherent with raw waveforms. Mixing supervision improves M2L's *latent geometry* (l_lat, mix_rate, sdr_dd all improve) but a consistency decoder can't turn that into clean waveform arithmetic the way the v2 GAN decoder does (v2.2-sym subtraction **+6.04** vs M2L Phase 2 **−11.18** dB vs GT). → Frame the v2/v3 GAN-decoder result as the headline; M2L as "recipe generalizes, decoder class caps the payoff."

4. **Phase 2b = Pareto illustration.** Over-weighting ℒ_dec (w=5) makes the decoder phase-deterministic (sdr_lin_gt −4.62 best, sdr_rec −0.45 best) but wrecks FAD (0.29) and decode-vs-decode sdr_lin (5.66). Quality-vs-phase-coherence trade-off — a figure, not a defect.

**Still pending before numbers freeze:** v2/v3 mixing re-eval with `sdr_lin_gt` (checkpoints exist), v3.0-baseline-d64 training (~35 h) + its evals, alpha sweep (`scripts/run_alpha_sweep.py`), v2/v3 subtraction `sdr_dd` re-run.

---

# v3 — from-scratch + higher compression

After v2 we wanted to address two reviewer-level concerns: (1) is 7.66× compression too loose for the linearity claim to mean anything, and (2) does the mixing-supervision recipe work *from scratch* rather than only as a 25k-step fine-tune over a strong v1.1 baseline? **v3.1** answers both: same v2.2-sym recipe (ℒ_dec=0.5, symmetric disc-on-mix), `d_model: 128 → 64` (compression 7.66× → 15.31×), trained 250k steps from random init.

## v3 + headline comparison

| Run | Compression | Init | FAD ↓ | sdr_rec ↑ | sdr_lin ↑ | l_lat ↓ | mix_rate ↓ |
|---|---|---|---|---|---|---|---|
| v2.0 control | 7.66× | warm-start | 0.0451 | +11.33 | +12.00 | 0.0678 | 1.203 |
| v2.2-sym (warm-start) | 7.66× | warm-start | **0.0440** | +11.38 | **+15.87** | **0.0493** | 1.054 |
| **v3.1** (new) | **15.31×** | **from scratch** | 0.0609 | +9.93 | **+14.64** | 0.0567 | **1.046** |

Files: [v3.1-decmix-disc-d64_mixing.json](v3.1-decmix-disc-d64_mixing.json), [v3.1-decmix-disc-d64_fad.json](v3.1-decmix-disc-d64_fad.json).

### Reading v3.1 vs v2.2-sym

- sdr_lin: −1.23 dB (14.64 vs 15.87). At 2× tighter compression and no warm-start, v3.1 retains 92% of v2.2-sym's mixing-equivariance gain.
- sdr_rec: −1.45 dB (9.93 vs 11.38). Compression tax on plain reconstruction — small for a 2× ratio change.
- FAD: 0.0609 vs 0.0440 (38% higher). Also explained by compression; v3.1's per-source FAD (fma 0.059 / maestro 0.100 / musdb 0.080) follows the same source-difficulty order as v2.x.
- l_lat: 0.057 vs 0.049 (slightly worse encoder linearity).
- mix_rate: 1.046 vs 1.054 — **slightly better** mixing-equivariance ratio than v2.2-sym.

### Bottom line for v3.1

Mixing-equivariance recipe is **not** an artifact of (a) the 7.66× compression rate or (b) the v1.1 warm-start. Same recipe, 2× tighter bottleneck, from random init, lands within 1.23 dB of v2.2-sym on sdr_lin. **+2.64 dB above v2.0 (the no-mixing baseline at the easier compression)**, so even at 15.31× from scratch we beat the no-mixing model at 7.66×.

### v3.0 control — TRAINED (June 2026), v3 gate PASSES

v3.0 (from-scratch d_model=64, 15.31×, NO mixing) trained 250k steps as the matched control for v3.1. Full-test mixing metrics (with sdr_lin_gt):

| metric | v3.0 (no mix) | v3.1 (mix) | Δ |
|---|---|---|---|
| sdr_rec | 9.98 | 9.93 | ~0 (free) |
| sdr_lin (dd) | 10.62 | 14.64 | +4.02 |
| **sdr_lin_gt** | 6.83 | **8.56** | **+1.73** |
| l_lat ↓ | 0.078 | 0.057 | better |
| mix_rate ↓ | 1.187 | 1.046 | better |
| FAD ↓ | (see v3.0_fad.json) | 0.061 | — |

**Gate verdict:** at 15.31× from scratch, mixing supervision adds **+1.73 dB ground-truth equivariance**, lower l_lat, and mix_rate 1.19→1.05, at **identical reconstruction** (9.98 vs 9.93). Because v3.0 is a matched control (same arch/compression/schedule, mixing off), this isolates the mixing-supervision contribution from compression cost — the gap the old "missing control" note flagged is now closed. Confirms the recipe is not an artifact of the 7.66× rate or warm-starting.

---

# Stem-removal downstream eval

Full MUSDB18-HQ test split (49 tracks after PR - Oh No skipped for stem-identity < 20 dB), 30 random non-silent 1-second windows per track, 1470 chunks × 4 stems = 5880 evaluations per model. For each (chunk, stem-to-remove):

```
z_full = f(x_mixture);  z_stem = f(x_stem)
x_hat  = g(z_full − z_stem)           # latent subtraction
target = x_mixture − x_stem            # ground truth (sum of remaining stems)
x_ceil = g(f(target))                  # encode-decode ceiling per model
```

We report SI-SDR of `x_hat` vs target, ceiling SI-SDR of `x_ceil` vs target, and **gap = ceil − sub** (the "linearity tax" per model: how much the AE can reconstruct vs how much the latent-arithmetic recovers).

Files: [v2.0-continued_subtraction.json](v2.0-continued_subtraction.json), [v2.2-decmix-disc_subtraction.json](v2.2-decmix-disc_subtraction.json), [v3.1-decmix-disc-d64_subtraction.json](v3.1-decmix-disc-d64_subtraction.json), [m2l-phase2_subtraction.json](m2l-phase2_subtraction.json).

## Headline table — per-stem SI-SDR (dB), higher is better

| Model | Compression | drums | bass | vocals | other | **all** | **gap (ceil − sub)** |
|---|---|---|---|---|---|---|---|
| v2.0-continued (no mixing) | 7.66× | +4.22 | +0.39 | +5.17 | +3.88 | **+3.42** | +5.21 |
| **v2.2-sym (mixing, warm-start)** | 7.66× | **+6.22** | +5.39 | **+7.51** | **+5.05** | **+6.04** | +2.60 |
| **v3.1 (mixing, from-scratch, 15×)** | 15.31× | +5.36 | **+4.73** | +6.85 | +3.87 | **+5.20** | **+2.47** |
| M2L Phase 2 (mixing, ~64×) | ~64× | −9.94 | −12.31 | −9.97 | −12.72 | **−11.23** | +8.41 |

(Bold = best across models on each cell; v3.1 / v2.2-sym share most cells.)

## Per-model ceiling (encode-decode of target waveform)

For reference — the AE's reconstruction quality on the residual targets (i.e. what's possible if you encode-decode without using latent arithmetic):

| Model | drums | bass | vocals | other | all |
|---|---|---|---|---|---|
| v2.0 | +9.17 | +7.26 | +9.64 | +8.43 | +8.62 |
| v2.2-sym | +9.20 | +7.29 | +9.67 | +8.38 | +8.64 |
| v3.1 | +7.97 | +6.29 | +8.80 | +7.63 | +7.67 |
| M2L Phase 2 | −2.31 | −3.57 | −1.80 | −3.62 | −2.82 |

v2.0 and v2.2-sym have essentially identical ceilings (same architecture, same compression). v3.1 sits ~1 dB lower (15.31× compression tax). M2L's ceiling is negative because M2L's overall sdr_rec on the test set is already negative (≈ −2.60 dB) — the residual targets inherit that.

## Key findings

1. **v3.1 is essentially at v2.2-sym's level on latent subtraction.** All-stem subtraction: 5.20 vs 6.04 dB, −0.84 dB at 2× tighter compression. The mixing-equivariance recipe carries the latent-arithmetic capability across the compression jump.

2. **v3.1's linearity gap is *smaller* than v2.2-sym's** (+2.47 vs +2.60 dB). The mixing-supervision pressure scales: a tighter latent forces less wasted-on-redundancy structure, and the mixing-supervision still aligns it linearly relative to its own ceiling.

3. **Bass is the dramatic stem.** v2.0 essentially fails on bass (+0.39 dB), while both v2.2-sym (+5.39) and v3.1 (+4.73) recover cleanly. **+4.34 dB v3.1 over v2.0** on bass alone — the strongest single-stem signal that mixing-equivariance is doing what we think.

4. **M2L's latent space doesn't support clean stem subtraction.** M2L Phase 2's subtraction crashes to −11.23 dB despite its ceiling being only −2.82 dB. The linearity tax is +8.41 dB — 3× larger than the waveform AEs. STFT-domain + consistency-decoder architecture gains on the `sdr_lin` mixing metric (+1.89 dB Phase 2 vs Phase 0) but does **not** translate that to clean latent arithmetic at this compression rate.

5. **"Other" is the hardest stem for all models.** It's the catch-all in MUSDB (guitars, keys, FX, anything not drums/bass/vocals). The within-class variability of "other" makes its target noisier than the named instruments.

## Reproducibility

| File | Purpose |
|---|---|
| [evaluation/compute_subtraction.py](../compute_subtraction.py) | Formal eval script, v2/v3 checkpoints, JSON output |
| [evaluation/m2l_run_subtraction.py](../m2l_run_subtraction.py) | Wraps compute_subtraction for M2L via [m2l_adapter.py](../m2l_adapter.py) |
| [scripts/run_subtraction.py](../../scripts/run_subtraction.py) | Driver across all 4 checkpoints, writes JSONs into this folder |
| `dataset/musdb18/test/` | 50-track test split (49 used; PR - Oh No excluded for stem-identity 18.5 dB < 20 threshold) |

Repro command (all four runs):

```
python -m scripts.run_subtraction
```

Default 30 chunks/track at seed=0; deterministic across reruns.
