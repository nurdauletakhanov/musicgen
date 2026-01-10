# Mixing-Equivariant STFT Autoencoder

A research prototype for learning 1-second audio representations whose latent space preserves **linear mixing** relationships.

This project implements an **STFT-based autoencoder** that compresses a 1-second audio chunk into a compact latent representation, while enforcing the property:

\[
\text{enc}(\lambda x_1 + (1-\lambda) x_2) \approx \lambda \cdot \text{enc}(x_1) + (1-\lambda) \cdot \text{enc}(x_2)
\]

This *mixing-equivariance constraint* improves controllability, interpolation, and downstream generative modeling by aligning the latent space with the linear structure of the audio domain (which is physically linear for sound pressure).

---

## **Project Goals**

### 1. Build an autoencoder for 1-second audio chunks

* STFT input (44.1 kHz sample rate, 1024-point FFT)
* Conv2D segment encoder → Transformer → latent tokens → Conv1D decoder → waveform

### 2. Enforce **mixing linearity**

Given audio chunks `x1`, `x2`, and mixture `xλ = λ x1 + (1−λ)x2`, the autoencoder should satisfy:

* **Latent linearity**
  \[
  f(x_\lambda) \approx \lambda f(x_1) + (1 - \lambda) f(x_2)
  \]

* **Decode mixing** (recommended)
  \[
  D(\lambda z_1 + (1-\lambda) z_2) \approx \lambda x_1 + (1-\lambda) x_2
  \]

### 3. Provide a clean benchmark for comparing audio autoencoders

* Reconstruction metrics: MR-STFT, log-magnitude loss, L1 waveform
* Latent space metrics: linearity error, decode mixing rate
* Ablations: segment lengths, d_model sizes, loss weight combinations

---

## **Architecture Overview**

### **1. STFT Preprocessing**

Each 1-second waveform is converted to a complex STFT:

```
waveform [B, 1, L] → STFT [B, 2, F, T]
- F = n_fft // 2 + 1 = 513 frequency bins
- T = L // hop_length = 172 frames (at 44.1kHz, hop=256)
- 2 channels: real and imaginary parts
```

### **2. Segment Encoder (Conv2D)**

The STFT is divided into `num_segments` temporal segments. Each segment is encoded using Conv2D layers that treat frequency and time as spatial dimensions:

```
[B, num_segments, 2, F, T_seg] → Conv2D stack → [B, num_segments, d_model]
```

This architecture respects frequency locality (harmonics are adjacent bins) and uses GroupNorm for training stability.

### **3. Global Encoder (Transformer)**

A Transformer with positional embeddings models the sequence of segment embeddings:

```
input:  [token_1, token_2, ... token_S]  (S = num_segments)
output: z ∈ ℝ^{S × d_model}
```

This produces **S latent tokens** representing the 1-second chunk.

### **4. Decoder (Conv1D Upsampling)**

Maps the latent tokens back into a waveform using progressive upsampling:

```
z [B, S, d_model] → project → [B, C, S] → UpsampleBlocks → [B, 1, L]
```

Each UpsampleBlock uses interpolation + Conv1D + dilated ResBlocks for high-quality waveform synthesis.

## **Loss Functions**

### **1. Reconstruction Loss**

Combination of MR-STFT + L1 waveform:

\[
\mathcal{L}_\text{recon}(x, \hat{x}) = \alpha \cdot \text{MRSTFT}(x, \hat{x}) + \beta \cdot |x - \hat{x}|_1
\]

MR-STFT uses multiple FFT sizes (512, 1024, 2048) for spectral convergence and log-magnitude losses.

### **2. Latent Mixing Loss** (optional)

For randomly sampled `(x1, x2)` and λ ∼ Uniform(0,1):

\[
\mathcal{L}_\text{latent-mix} = \big\| E(x_\lambda) - [\lambda E(x_1) + (1-\lambda) E(x_2)] \big\|_2^2
\]

where \(x_\lambda = \lambda x_1 + (1-\lambda) x_2\).

### **3. Decode Mixing Loss** (recommended)

Compares latent interpolation vs. real autoencoder on mixed input:

\[
\mathcal{L}_\text{decode-mix} = \text{MRSTFT}(D(\lambda z_1 + (1-\lambda) z_2), x_\lambda) + |D(\lambda z_1 + (1-\lambda) z_2) - x_\lambda|_1
\]

A "rate" metric tracks how close interpolation is to real encoding (ideal = 1.0).

### **Total Loss**

\[
\mathcal{L} = \mathcal{L}_\text{recon} + \gamma \mathcal{L}_\text{latent-mix} + \delta \mathcal{L}_\text{decode-mix}
\]

---

## **Why This Project is Interesting**

### **Most existing audio autoencoders:**

* Work with 10–40 ms windows (Encodec, DAC, SoundStream).
* Produce long sequences of tokens.
* Are not designed to preserve the linear structure of audio.
* Are optimized mainly for perceptual quality and compression.

### **This project attempts something new:**

1. **A 1-second representation** (much more compressed than typical neural codecs).
2. **Mixing-equivariance as a core constraint** (latent space respects physical mixing).
3. **STFT-based encoding with waveform output** (efficient frequency representation, direct audio synthesis).
4. **Compatibility with simple latent generative models** (AR, diffusion).
5. **New metrics** for evaluating linearity in audio embeddings (decode mixing rate).

This could open new avenues for:

* Audio interpolation and morphing
* Style transfer
* Layered music creation (blending stems)
* Generative models that obey physical mixing rules
* Compact representations for downstream music models

---

## **Repository Structure**

```
musicgen/
│
├── models/
│   ├── autoencoder.py    # Main autoencoder with loss computation
│   ├── encoder.py        # Conv2D segment encoder + Transformer
│   └── decoder.py        # Conv1D upsampling decoder
│
├── data/
│   ├── dataloader.py     # STFTChunkDataset + ShardedSampler
│   └── preprocess.py     # STFT preprocessing for MAESTRO
│
├── training/
│   ├── train.py          # Training loop and Trainer class
│   ├── utils.py          # Logging, checkpointing utilities
│   └── hub_utils.py      # HuggingFace Hub integration
│
├── evaluation/
│   └── reconstruct.py    # Reconstruction evaluation scripts
│
├── scripts/
│   └── upload_to_hub.py  # Upload checkpoints to HuggingFace
│
├── checkpoints/          # Saved model checkpoints
├── config.yaml           # Training configuration
├── requirements.txt
└── README.md
```

---

## **Example Usage**

### **Forward pass**

```python
from models.autoencoder import Autoencoder

model = Autoencoder(
    d_model=256,
    n_heads=8,
    n_layers=6,
    num_segments=25,
    n_freq_bins=513,           # n_fft // 2 + 1
    channels=[512, 512, 256, 128, 64],
    upsampling_factors=[6, 6, 7, 7],
    target_length=44100,       # 1 second at 44.1kHz
)

# Input: STFT [B, 2, F, T] and waveform [B, 1, L]
x_stft = torch.randn(8, 2, 513, 172)
x_wave = torch.randn(8, 1, 44100)

loss, components = model(x_stft, x_wave)
print(components)  # {'total': ..., 'recon': ..., 'wav_l1': ..., 'mrstft': ..., ...}
```

### **Latent mixing**

```python
# Encode two different audio chunks
z1 = model.encoder(x_stft_1)  # [B, num_segments, d_model]
z2 = model.encoder(x_stft_2)

# Interpolate in latent space
lam = 0.5
z_mix = lam * z1 + (1 - lam) * z2

# Decode the interpolated latent
x_mix = model.decoder(z_mix)  # [B, 1, target_length]
```

---

## **Research Experiments**

### **Phase A: Baseline** ✓

* Train the autoencoder without mixing loss.
* Evaluate reconstruction quality (MR-STFT, L1).

### **Phase B: Add mixing-equivariant loss** ✓

* Add decode mixing loss for latent space linearity.
* Measure linearity error across λ ∈ [0,1].
* Track decode mixing rate (ideal = 1.0).

### **Phase C: Architecture tuning**

* Ablate num_segments, d_model, loss weights.
* Optimize for reconstruction + linearity tradeoff.

### **Phase D: Generation**

* Train a small AR or diffusion model over latent tokens.
* Evaluate interpolation and style blending.

---

## **Installation**

```
pip install -r requirements.txt
```

---

## **HuggingFace Hub Integration**

This project supports uploading and downloading model checkpoints to/from HuggingFace Hub for easy sharing, versioning, and remote storage.

### **Authentication**

First, authenticate with HuggingFace Hub:

```bash
# Option 1: Using CLI (recommended)
huggingface-cli login

# Option 2: Set environment variable
export HUGGINGFACE_HUB_TOKEN=your_token_here

# Option 3: Programmatically (in Python)
from training.hub_utils import authenticate
authenticate(token="your_token_here")
```

Get your token from: https://huggingface.co/settings/tokens

### **Configuration**

Enable HuggingFace Hub integration in your `config.yaml`:

```yaml
huggingface:
  enabled: true                    # Toggle Hub integration
  repo_id: "username/model-name"   # Repository ID (or auto-generated from config name)
  push_best: true                  # Upload best_model.pth automatically
  push_checkpoints: false          # Upload regular checkpoints (can be expensive)
  push_interval: 5                 # Upload every N epochs if push_checkpoints=True
  private: false                   # Create private repository if it doesn't exist
```

If `repo_id` is not specified, it will be auto-generated from the config name as `{username}/{name}-autoencoder`.

### **Automatic Upload During Training**

When enabled, checkpoints are automatically uploaded to the Hub during training:

- `best_model.pth` is uploaded automatically when a new best model is saved (if `push_best: true`)
- Regular checkpoints are uploaded according to `push_interval` (if `push_checkpoints: true`)

The training config file is also uploaded once at initialization.

### **Loading Models from Hub**

You can load checkpoints directly from HuggingFace Hub using repository IDs:

```python
from training.train import Trainer

# Load checkpoint from Hub
trainer = Trainer("config.yaml")
trainer.build_model()
start_epoch, best_val_loss = trainer.load_checkpoint("username/model-name")
```

Or in evaluation scripts:

```python
# In evaluation/reconstruct.py
# Use Hub ID instead of local path
python evaluation/reconstruct.py --config evaluation/config.yaml
# In config.yaml, set checkpoint: "username/model-name"
# The script will automatically download from Hub if not found locally
```

Supported checkpoint path formats:
- Local path: `checkpoints/stft-vocoder/best_model.pth`
- Hub ID: `username/model-name` (loads `best_model.pth` from repo)
- Hub path: `username/model-name/checkpoints/checkpoint_10.pth` (specific file)
- Hub prefix: `hub://username/model-name` (explicit Hub identifier)

### **Uploading Existing Checkpoints**

Upload checkpoints that were trained before Hub integration:

```bash
# Upload a single checkpoint
python scripts/upload_to_hub.py \
    --checkpoint checkpoints/stft-vocoder/best_model.pth \
    --repo-id username/model-name

# Upload all checkpoints from a directory
python scripts/upload_to_hub.py \
    --checkpoint-dir checkpoints/stft-vocoder \
    --repo-id username/model-name \
    --upload-all

# Upload config file as well
python scripts/upload_to_hub.py \
    --checkpoint checkpoints/stft-vocoder/best_model.pth \
    --repo-id username/model-name \
    --config config.yaml
```

### **Repository Structure on Hub**

Uploaded repositories follow this structure:

```
username/model-name/
├── README.md                    # Model card (auto-generated)
├── config.yaml                  # Training configuration
├── best_model.pth              # Best checkpoint
└── checkpoints/                 # Regular checkpoints (if enabled)
    ├── checkpoint_10.pth
    ├── checkpoint_20.pth
    └── ...
```

### **Listing Available Checkpoints**

List all checkpoints available in a Hub repository:

```python
from training.hub_utils import list_hub_checkpoints

checkpoints = list_hub_checkpoints("username/model-name")
print(checkpoints)
# ['best_model.pth', 'checkpoints/checkpoint_10.pth', ...]
```

### **Troubleshooting**

**Authentication errors:**
- Make sure you've run `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`
- Verify your token has write permissions at https://huggingface.co/settings/tokens

**Upload failures:**
- Check your internet connection
- Verify repository exists and you have write access
- Check file size limits (free tier has 10GB limit per repository)

**Download failures:**
- Verify the repository ID and filename are correct
- Check if the repository is private and you have access
- Ensure you have sufficient disk space

---

## **Status: Research Prototype**

This repository is not yet optimized for production.
Training stability, decoder design, and mixing-equivariance are active research topics.

Contributions and discussions are welcome.
