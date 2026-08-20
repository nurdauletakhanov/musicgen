# Reproducing the paper

Maps every table and figure in *"An Explicit Decode-Mixing Loss for
Mixing-Equivariant Audio Autoencoders"* to the command that produces it.

**No number in the paper is hand-typed.** Every table cell is generated from
the JSONs in [`evaluation/v2_metrics/`](evaluation/v2_metrics/) by
[`research/paper/icassp2027/make_tables.py`](research/paper/icassp2027/make_tables.py).
So there are two levels of reproduction, and you can stop at whichever you need:

| Level | What you verify | What you need | Cost |
|---|---|---|---|
| **1. Paper from metrics** | tables/figures match the recorded metrics | this repo only | seconds |
| **2. Metrics from weights** | the metrics match the trained models | + checkpoints + datasets | hours |
| **3. Weights from scratch** | the models match the configs | + datasets + a GPU | ~35 h/run |

---

## Level 1 — rebuild the paper from the recorded metrics

Needs nothing but this repo and a LaTeX install.

```bash
cd research/paper/icassp2027
python make_tables.py      # tables/*.tex   <- evaluation/v2_metrics/*.json
python make_figures.py     # figures/*.pdf  <- alpha_sweep_summary.json + diag log
latexmk -g -pdf paper.tex
```

`make_tables.py` regenerating byte-identical `.tex` files is itself the check
that the paper matches the recorded data.

> If you change figures, re-verify no Type 3 fonts crept in — IEEE PDF eXpress
> rejects them:
> `gs -q -dNODISPLAY -dBATCH -dNOPAUSE -dPDFINFO paper.pdf | grep -E "Type3|Not embedded"`
> (empty output = pass).

## Level 2 — regenerate the metrics from checkpoints

Needs the preprocessed test set (see the README's Data section) and the
checkpoints from
[`SoMa25/mixing-equivariant-ae-checkpoints`](https://huggingface.co/SoMa25/mixing-equivariant-ae-checkpoints),
placed at `checkpoints/<run>/best.pth`.

| Paper element | Command | Writes |
|---|---|---|
| **Table I** (mechanism ablation) | `python -m scripts.run_v2_mixing_metrics`<br>`python -m scripts.run_v2_fad` | `v2.*_mixing.json`, `v2.*_fad.json` |
| **Table II** (compression + architecture) | same, plus `--only v3.0-baseline-d64 v3.1-decmix-disc-d64` | `v3.*_{mixing,fad}.json` |
| **Table III** (per-domain equivariance) | same JSONs as I/II — per-source keys are already in them | — |
| **Table IV** (stem subtraction) | `python -m scripts.run_subtraction` | `*_subtraction.json` |
| **Fig. 1** (decode-noise protocol confound) | `python -m scripts._diag_old_vs_new_eval` | `_diag_old_vs_new_eval.log` |
| **Fig. 3** (α sweep) | `python -m scripts.run_alpha_sweep` | `alpha_sweep/`, `alpha_sweep_summary.json` |

Every driver skips already-existing outputs unless `--force` is passed, and
every one takes `--only <run> [...]` to run a subset.

**Smoke-test first.** Each driver has a fast mode that exercises the whole path
in minutes rather than hours:

```bash
python -m scripts.run_v2_mixing_metrics --max-batches 5
python -m scripts.run_v2_fad            --max-tracks-per-source 3
python -m scripts.run_subtraction       --max-tracks 5
python -m scripts.run_alpha_sweep       --max-batches 5
```

### Music2Latent rows

The M2L rows of Table II and Table IV need the **separate, non-public**
`music2latent-mix` repo. The eval adapter and all resulting JSONs are in this
repo, so the numbers are inspectable and the protocol is auditable, but the
fine-tuning runs cannot be repeated from here. Point the scripts at a local
checkout with:

```bash
export MUSICGEN_M2L_REPO=/path/to/music2latent-mix
export MUSICGEN_M2L_PUBLISHED=/path/to/music2latent.pt      # Pasini et al. release
export MUSICGEN_M2L_CHECKPOINT=/path/to/phase2_ema.pt       # our fine-tune
```

`scripts/run_m2l_phase3.py` drives the EMA-consistent re-eval of all M2L phases.

## Level 3 — retrain from scratch

```bash
python -m training.train --config configs/experiments/v2/v2.1_decmix.yaml
```

Configs for all eleven runs are under [`configs/experiments/`](configs/experiments/).
The v1/v3 runs are 250k steps (~35 h on a single RTX 5090); the v2.x runs are
25k-step fine-tunes from `v1.1`, so they need `checkpoints/v1.1/best.pth` first.

**Expect seed variance.** All published numbers are n=1. A rerun landing within
a few tenths of a dB on SDR_lin_gt is a match; do not read a 0.3 dB difference
between v2.1 and v2.2 as meaningful — the paper does not.

---

## Metric definitions

Implemented in [`evaluation/compute_mixing_metrics.py`](evaluation/compute_mixing_metrics.py).
For a pair (x₁, x₂), z̄ = α·f(x₁) + (1−α)·f(x₂) and x̄ = α·x₁ + (1−α)·x₂:

| Metric | Definition | Notes |
|---|---|---|
| `sdr_rec` | SI-SDR(g(f(x)), x) | plain reconstruction |
| `sdr_lin` | SI-SDR(g(z̄), g(f(x̄))) | **decode-vs-decode.** Gameable — see below |
| `sdr_lin_gt` | SI-SDR(g(z̄), x̄) | vs ground truth; the metric of record |
| `l_lat` | ‖z̄ − f(x̄)‖² / ‖f(x̄)‖² | encoder linearity (normalized) |
| `mix_rate` | L_recon(g(z̄), x̄) / L_recon(g(f(x̄)), x̄) | decoder equivariance; <1 beats encode-then-decode |
| `gap` | ceiling − subtraction SI-SDR | "linearity tax" vs the model's own ceiling |
| `fad` | Fréchet Audio Distance | LAION-CLAP backbone, 10 s clips @ 48 kHz |

Two traps worth knowing before you compare against other papers:

1. **`sdr_lin` is not comparable across papers and is gameable.** It measures
   consistency between two decode paths, not equivariance. A decoder pulled onto
   a common realistic manifold raises it without moving g(z̄) toward x̄ — exactly
   what the discriminator-on-mix ablation does (12.1 → 16.0 dB with *no*
   ground-truth gain). Use `sdr_lin_gt`.
2. **Stochastic decoders need shared decode noise.** A consistency-model decoder
   draws fresh noise per call, which sets output phase. Two decodes of the same
   latent then differ in phase and `sdr_lin` collapses to ≈ −8 dB regardless of
   real linearity. All evals here share the noise across the two decode calls;
   `scripts/_diag_old_vs_new_eval.py` demonstrates the ≈12 dB swing between
   protocols on identical checkpoints.
