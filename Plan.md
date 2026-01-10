# Linearity in Audio Latent Spaces — Research Plan

## 1. Goal

Study **latent linearity in audio autoencoders**, identify when it emerges naturally,
when it fails, and whether enforcing linearity improves **latent diffusion models**.

Core questions:
- Is latent linearity an artifact of *additive datasets* (e.g. piano)?
- Does linearity break on multi-instrument / nonlinear audio?
- Does enforcing linearity improve downstream latent diffusion?

---

## 2. Hypotheses

H1. On additive harmonic datasets (e.g. piano), **standard autoencoders already exhibit near-linear latent interpolation**.

H2. On multi-instrument or nonlinear audio, **latent linearity degrades significantly**.

H3. Explicit linearity constraints in the autoencoder **improve interpolation fidelity** and **latent diffusion stability**.

---

## 3. Datasets

### 3.1 Easy (Additive) Dataset
- MAESTRO (piano-only)
- Purpose: show that linearity can emerge *without* explicit constraints

### 3.2 Hard (Nonlinear / Multi-source) Dataset (choose ≥1)
- NSynth (multi-instrument notes)
- Optional extensions:
  - speech (subset)
  - piano + effects (reverb, saturation)
  - simple mixtures (speech + music)

---

## 4. Models

### 4.1 Autoencoders

| Model | Description |
|-----|------------|
| AE-Baseline | Standard autoencoder (no linearity loss) |
| AE-Linear | Same architecture + decode mixing loss (and optional latent mix loss) |

Constraints:
- Same architecture
- Same training setup
- Only difference is linearity loss

### 4.2 External Baseline (Optional but Recommended)

Use **one frozen pretrained neural audio codec**:
- Evaluate linearity *without retraining*
- Test interpolation in continuous latent space (pre-quantization if available)

Purpose:
- Show whether linearity behavior is model-specific or dataset-driven

---

## 5. Metrics

### 5.1 Reconstruction
- ReconSingle: D(E(x)) vs x

### 5.2 Mixing Evaluation (Core Contribution)

Given:
- x_mix = α x₁ + (1 − α) x₂
- z₁ = E(x₁), z₂ = E(x₂)

Compute:

- ReconMixReal = loss(D(E(x_mix)), x_mix)
- ReconMixInterp = loss(D(α z₁ + (1 − α) z₂), x_mix)

Metrics:
- **MixInterp Loss** (absolute quality)
- **MixReal Loss** (AE oracle)
- **Rate = MixInterp / MixReal**

Important:
- Report **absolute losses**, not only rate
- Rate alone is insufficient

### 5.3 α-Sweep Evaluation
Evaluate metrics for:
α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}

---

## 6. Experimental Protocol

### Phase 1 — Dataset Effect
1. Train AE-Baseline on MAESTRO
2. Measure linearity metrics
3. Repeat on hard dataset
4. Observe degradation

### Phase 2 — Linearity Constraint
1. Train AE-Linear on same datasets
2. Compare:
   - MixInterp loss
   - Rate
   - Failure cases (α extremes)

### Phase 3 — External Baseline (Optional)
1. Evaluate pretrained codec on same tests
2. Compare linearity behavior across datasets

---

## 7. Diffusion Sanity Check (Downstream Validation)

### Setup
- Train identical latent diffusion models on:
  - AE-Baseline latents
  - AE-Linear latents

### Evaluation (qualitative + simple metrics)
- Sampling stability
- Interpolation robustness
- Off-manifold drift
- Consistency of generated mixes

Diffusion is **not the main contribution** — only a validation signal.

---

## 8. Expected Outcomes

- MAESTRO:
  - High linearity even for AE-Baseline
- Hard dataset:
  - Linearity degrades sharply
- AE-Linear:
  - Improves interpolation fidelity
  - Improves diffusion robustness

---

## 9. Paper Structure (Draft)

1. Introduction
2. Why Audio Linearity Is Dataset-Dependent
3. Linearity Metrics and Evaluation Protocol
4. Empirical Results
   - MAESTRO vs Hard Dataset
   - Baseline vs Linear AE
5. Diffusion Case Study
6. Limitations
7. Conclusion

---

## 10. Scope Control (Important)

Not included:
- New diffusion architectures
- Human listening studies
- SOTA audio benchmarks

Focus:
- Controlled phenomenon study
- Clear causal conclusions
