# Plan: Research Strategy for Mixing-Equivariant Latent Spaces Paper

## Context

We have a working mixing-equivariant STFT autoencoder with strong preliminary results (MixRate@0.5 = 0.981 vs EnCodec 1.292, DAC 1.358). A competing paper — Torres et al., "Learning Linearity in Audio Consistency Autoencoders via Implicit Regularization" (ICASSP 2026) — achieves similar goals via data augmentation (implicit regularization) on Music2Latent. We need to differentiate, choose a venue, and plan experiments.

## Venue Decision: **ICLR 2027** (~October 2026 deadline)

**Why ICLR 2027 (A\*):**
- ~7 months gives time to build the full story: autoencoder + MixRate metric + Pareto frontier + latent diffusion + compositional generation
- ICLR values representation learning — "structured latent spaces improve downstream generation" is core ICLR material
- Torres et al. becomes cited prior work (not awkward concurrent work)
- 8 pages + appendix = enough room for the full contribution

**Why not other venues:**
- ISMIR 2026 (Apr 27): Too tight, autoencoder-only story is weakened by Torres et al.
- NeurIPS 2026 (~May/Jun): Only 2-3 months, rushed diffusion story is risky
- ICML 2027 (~Jan 2027): Fallback if ICLR doesn't work out

## Differentiation from Torres et al.

| Aspect | Torres et al. (ICASSP 2026) | Ours |
|--------|---------------------------|------|
| Method | Implicit (data augmentation) | **Explicit decode-mixing loss** (controllable λ_mix) |
| Metric | Standard (SNR, MSS, SI-SDR) | **MixRate** (novel, transferable) |
| Mixing | Equal-weight only (0.5+0.5) | **Arbitrary α ∈ [0,1]** with alpha sweep |
| Architecture | Music2Latent (consistency AE, U-Net) | **STFT-native 2D conv + Transformer** |
| Application | Oracle source separation | **Compositional latent diffusion generation** |
| Trade-off | No control (implicit) | **Pareto frontier** via λ_mix ablation |
| Compression | 64x only | **Multiple regimes** (9x, 24x, 46x) |

**Key framing:** "Explicit mixing equivariance enables compositional latent diffusion — generating stems independently and composing them via linear latent arithmetic."

## Experiment Roadmap

**Key constraint:** GPU training time is the bottleneck (~12 hours per 30-epoch Phase 2 run). Coding is fast with Claude Code. Strategy: parallelize — run training on GPU while coding diffusion.

### Track A: Autoencoder Sweep (April 2026, ~2 weeks GPU)

1. **Expand λ_mix sweep** — Add configs for λ_mix = {1.0, 2.0} to find the breakdown point
   - Create: `configs/experiments/compression/comp_24x_v6_phase2_mix1.0.yaml`, `...mix2.0.yaml`
   - Train: 40 epochs each from Phase 1 checkpoint (~12h each)
   - Currently running: λ_mix=0.5. Queue 1.0 and 2.0 after.
   - Full sweep: {0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0}

2. **Evaluate all sweep checkpoints** — Run `test_evaluation.py` + `select_checkpoint.py` on each
   - Scripts: `evaluation/test_evaluation.py`, `evaluation/select_checkpoint.py`
   - Produces: Pareto frontier (reconstruction quality vs MixRate)

3. **Perceptual metrics** — Add FAD + SI-SDR to eval pipeline
   - Extend `evaluation/utils.py`
   - Run on all model variants + EnCodec + DAC

### Track B: Latent Diffusion (START NOW, parallel with Track A)

4. **Implement DiT-style denoising network** for z ∈ R^{96×96}
   - Self-attention over S=96 tokens with d=96 (5s audio, 24x compression model with GAN)
   - AdaLN conditioning on timestep + stem-type class embedding
   - DDPM or flow matching
   - New files: `models/diffusion.py`, `training/diffusion_trainer.py`
   - Architecture: comp-24x-v6 autoencoder (96 segments, d_model=96, discriminator-trained)

5. **Extract and cache latent dataset** — Encode all MUSDB18 5s chunks through frozen Phase 2 encoder

6. **Train latent diffusion** — Unconditional first, then stem-conditioned
   - Optional: encode FMA/Jamendo through frozen encoder for more training data (no autoencoder retraining needed)

### Track C: Compositional Generation Eval (after diffusion trains)

7. **Compositional generation experiments:**
   - Generate stems by type → mix in latent space → decode
   - Evaluate FAD, Compositional Coherence Score
   - Compare equivariant vs non-equivariant autoencoder latents

8. **Diffusion training efficiency** — Convergence curves: Phase 1 vs Phase 2 latents

9. **K-way mixing** — Test 4-way mixing (all MUSDB18 stems simultaneously)

### Track D: Strengthening (as time allows)

10. Per-stem-pair MixRate breakdown
11. Latent space visualization (t-SNE/UMAP)
12. VAE baseline (KL divergence instead of mixing loss)

## Paper Structure (ICLR 2027, 8pp + appendix)

**Title:** "Mixing-Equivariant Latent Spaces for Compositional Music Generation"

1. **Introduction** (1.5pp) — Audio mixing is linear, codecs break this, explicit equivariance enables compositional diffusion
2. **Related Work** (0.75pp) — Codecs, latent diffusion, equivariant representations (cite Torres et al.), compositional generation
3. **Method** (2pp) — Architecture, decode-mixing loss, MixRate metric, two-phase training, latent diffusion pipeline
4. **Experiments** (2.5pp) — Reconstruction quality, MixRate alpha sweep, λ_mix Pareto frontier, compression regimes, compositional generation, diffusion efficiency
5. **Discussion** (0.5pp) — Pareto interpretation, limitations (MUSDB18 scale, 4-stem), broader impact
6. **Conclusion** (0.25pp)
7. **Appendix** — Architecture details, per-stem analysis, visualizations, comparison with Torres et al.

**4 Figures:** Architecture diagram, Pareto frontier plot, alpha sweep curves, compositional generation pipeline
**4 Tables:** Main results (+FAD/SI-SDR), alpha sweep, λ_mix ablation, compositional generation quality

## Timeline (parallelized — code while GPU trains)

| Period | GPU (background) | Active work |
|--------|-----------------|-------------|
| Now - Apr 10 | λ_mix=0.5 finishes, start 1.0 | **Implement diffusion model** (DiT) |
| Apr 10-20 | λ_mix=1.0 trains, start 2.0 | Extract latent dataset, finish diffusion code |
| Apr 20-30 | λ_mix=2.0 trains | Evaluate all sweep checkpoints, add FAD/SI-SDR |
| May 1-15 | **Start diffusion training** (unconditional) | Pareto frontier analysis, compression eval |
| May 15-31 | Diffusion training continues | Add stem conditioning, iterate on diffusion |
| Jun 1-30 | Stem-conditioned diffusion trains | Compositional eval code, strengthening experiments |
| Jul 1-31 | Optional: more diffusion / data augmentation | Compositional generation experiments, K-way mixing |
| Aug 1-31 | Final experiments | **Write paper** (methods, experiments, results) |
| Sep 1-30 | -- | Write paper (intro, related work, discussion), revise |
| Oct 1-deadline | -- | Final polish + submit |

## Risks

1. **Diffusion quality poor on small MUSDB18** → Encode FMA/MTG-Jamendo through frozen encoder for more diffusion training data (no autoencoder retraining needed)
2. **Compositional mixing of generated stems sounds bad** → Document as finding, propose latent normalization solutions
3. **Torres et al. follow-up** → Our explicit loss + MixRate + Pareto frontier are distinct regardless
4. **Fallback:** If diffusion story incomplete, submit autoencoder paper to ICASSP 2027 (Sep deadline, 4pp) or ISMIR 2027

## Verification

- Run all `test_evaluation.py` results and compare against existing `results/` JSON files
- Listen to generated audio samples for qualitative validation
- Reproduce EnCodec/DAC MixRate baselines to confirm numbers
- For diffusion: evaluate FAD against MUSDB18 test set as reference distribution
