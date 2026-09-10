"""Upload/refresh the model card (README.md) on the HF weights repo.

Run anywhere you are logged in to Hugging Face as the repo owner:
  python -m scripts.upload_model_card
"""
import os

from huggingface_hub import HfApi

REPO_ID = os.environ.get("MUSICGEN_HF_REPO",
                         "SoMa25/mixing-equivariant-ae-checkpoints")

CARD = """---
license: mit
tags: [audio, autoencoder, music, representation-learning]
---

# Mixing-Equivariant Audio Autoencoder — checkpoints

Inference weights for **"What Makes Audio Latents Mixing-Equivariant? A Controlled
Study of Explicit Supervision"** (ICASSP 2027 submission).

Code, configs, eval scripts, and the paper source:
**https://github.com/nurdauletakhanov/musicgen** — see its `REPRODUCING.md`
for the exact command behind every number in the paper.

## Layout

- `musicgen/<run>/best.pth` — waveform GAN autoencoder runs (v1/v2/v3 lineage;
  33.5M params, 44.1 kHz mono). Load with the matching config under
  `configs/experiments/` in the code repo.
- `m2l/<phase>_ema.pt` — EMA-merged Music2Latent fine-tunes (cross-architecture
  experiments, paper Sec. IV-B).
- `MANIFEST.md` — file-by-file provenance.

## Quick use

```python
# in a checkout of the code repo
from training.config import load_config, build_model_config
from models.autoencoder import Autoencoder
import torch

cfg = load_config("configs/experiments/v2/v2.1_decmix.yaml")
model = Autoencoder(**build_model_config(cfg))
ck = torch.load("best.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
# Stem removal. Subtract in the latent space, then ADD the encoded silence:
# its coefficients (1, -1) sum to zero, so without f(0) the encoder's offset
# does not cancel and the score mostly reflects that offset (see below).
zero = torch.zeros_like(mix)
z = model.encoder(mix) - model.encoder(stem) + model.encoder(zero)
residual, _ = model.decoder(z)

# Raw baseline, for comparison only — NOT the recommended form:
# z_raw = model.encoder(mix) - model.encoder(stem)
```

`v2.1-decmix` is the paper's recommended recipe (decode-mixing loss only).

## Two things to know before comparing these models

**Latent subtraction needs the origin.** Subtraction has coefficients
(1, -1), which sum to zero, so the encoder's offset `f(0)` drops out. For an
affine encoder `f(x) = Ax + b` the identity `f(mix) - f(stem) + f(0) = f(res)`
is exact, so the corrected decode is the model's own reconstruction of the
residual. Raw subtraction scores therefore mostly reflect each model's latent
offset, not the training recipe; correct it before comparing. On a held-out
corpus of 240 recordings the correction removes 87-95% of the apparent
advantage of mixing supervision, leaving a small residue of +0.12 to
+0.42 dB rather than nothing:

```bash
python -m evaluation.compute_subtraction --origin-correct ...
```

**Decode-vs-decode SI-SDR is confounded for the M2L checkpoints.** A
consistency decoder draws fresh noise per call. Decoding the same latent twice
with independent noise scores -1.8 dB with no latent arithmetic at all, so
share the decode noise (the eval adapter does) or the metric measures sampling,
not linearity.
"""

api = HfApi()
api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                repo_id=REPO_ID, repo_type="model")
print(f"model card uploaded to https://huggingface.co/{REPO_ID}")
