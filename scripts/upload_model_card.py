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

Inference weights for **"An Explicit Decode-Mixing Loss for Mixing-Equivariant
Audio Autoencoders"** (ICASSP 2027 submission).

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
# stem removal: model.decoder(model.encoder(mix) - model.encoder(stem))
```

`v2.1-decmix` is the paper's recommended recipe (decode-mixing loss only).
"""

api = HfApi()
api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                repo_id=REPO_ID, repo_type="model")
print(f"model card uploaded to https://huggingface.co/{REPO_ID}")
