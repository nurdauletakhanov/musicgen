# Mixing-Equivariant STFT Autoencoder

A research prototype for learning 1-second audio representations whose latent space preserves **linear mixing** relationships.

This project implements an **STFT-based autoencoder** that compresses a 1-second audio chunk into a compact latent representation, while enforcing the property:

$$
\text{enc}(\lambda x_{1} + (1-\lambda) x_{2}) \approx \lambda \cdot \text{enc}(x_{1}) + (1-\lambda) \cdot \text{enc}(x_{2})
$$

This *mixing-equivariance constraint* improves controllability, interpolation, and downstream generative modeling by aligning the latent space with the linear structure of the audio domain (which is physically linear for sound pressure).

---

## **Project Goals**

### 1. Build an autoencoder for 1-second audio chunks

* STFT input (44.1 kHz sample rate, 1024-point FFT)
* Conv2D segment encoder → Transformer → latent tokens → Conv1D decoder → waveform

### 2. Enforce **mixing linearity**

Given audio chunks `x1`, `x2`, and mixture $x_{\lambda} = \lambda x_{1} + (1-\lambda)x_{2}$, the autoencoder should satisfy:

**Latent linearity:**

$$
f(x_{\lambda}) \approx \lambda f(x_{1}) + (1 - \lambda) f(x_{2})
$$

**Decode mixing** (recommended):

$$
D(\lambda z_{1} + (1-\lambda) z_{2}) \approx \lambda x_{1} + (1-\lambda) x_{2}
$$

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
output: z in R^(S × d_model)
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

$$
\mathcal{L}_{\text{recon}}(x, \hat{x}) = \alpha \cdot \text{MRSTFT}(x, \hat{x}) + \beta \cdot |x - \hat{x}|_{1}
$$

MR-STFT uses multiple FFT sizes (512, 1024, 2048) for spectral convergence and log-magnitude losses.

### **2. Latent Mixing Loss** (optional)

For randomly sampled `(x1, x2)` and $\lambda \sim \text{Uniform}(0,1)$:

$$
\mathcal{L}_{\text{latent-mix}} = \big\| E(x_{\lambda}) - [\lambda E(x_{1}) + (1-\lambda) E(x_{2})] \big\|_{2}^{2}
$$

where $x_{\lambda} = \lambda x_{1} + (1-\lambda) x_{2}$.

### **3. Decode Mixing Loss** (recommended)

Compares latent interpolation vs. real autoencoder on mixed input:

$$
\mathcal{L}_{\text{decode-mix}} = \text{MRSTFT}(D(\lambda z_{1} + (1-\lambda) z_{2}), x_{\lambda}) + |D(\lambda z_{1} + (1-\lambda) z_{2}) - x_{\lambda}|_{1}
$$

A "rate" metric tracks how close interpolation is to real encoding (ideal = 1.0).

### **Total Loss**

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \gamma \mathcal{L}_{\text{latent-mix}} + \delta \mathcal{L}_{\text{decode-mix}}
$$

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
├── configs/                      # Configuration files
│   ├── base.yaml                 # Shared defaults (model arch, STFT params)
│   └── experiments/              # Experiment-specific overrides
│       └── current.yaml          # Active experiment config
│
├── models/
│   ├── autoencoder.py            # Main autoencoder with loss computation
│   ├── encoder.py                # Conv2D segment encoder + Transformer
│   └── decoder.py                # Conv1D upsampling decoder
│
├── data/
│   ├── dataloader.py             # STFTChunkDataset + ShardedSampler
│   └── preprocess.py             # STFT preprocessing for MAESTRO
│
├── training/
│   ├── train.py                  # Training loop and Trainer class
│   ├── utils.py                  # Logging, checkpointing utilities
│   └── hub_utils.py              # HuggingFace Hub integration
│
├── evaluation/
│   └── reconstruct.py            # Audio reconstruction evaluation
│
├── scripts/
│   ├── upload_to_hub.py          # Upload checkpoints to HuggingFace
│   └── check_normalization.py    # Check data normalization statistics
│
├── checkpoints/                  # Saved model checkpoints (git-ignored)
│   ├── experiments.yaml          # Registry of all experiments (tracked)
│   └── <experiment-name>/        # Per-experiment checkpoints
│       ├── config.yaml           # Config used for this run
│       ├── best_model.pth
│       └── checkpoint_*.pth
│
├── .gitignore
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
* Measure linearity error across $\lambda \in [0, 1]$.
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

Enable HuggingFace Hub integration in your experiment config (`configs/experiments/current.yaml`):

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

# Load checkpoint from Hub (uses configs/experiments/current.yaml by default)
trainer = Trainer("configs/experiments/current.yaml")
trainer.build_model()
start_epoch, best_val_loss = trainer.load_checkpoint("username/model-name")
```

Or in evaluation scripts:

```bash
# In evaluation/reconstruct.py
# Use Hub ID directly as checkpoint path
python evaluation/reconstruct.py --checkpoint username/model-name
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
    --config configs/experiments/current.yaml
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

## **TensorBoard Integration**

This project includes comprehensive TensorBoard logging for monitoring training progress, visualizing metrics, and analyzing model performance. TensorBoard logs are automatically generated during training and provide detailed insights into model behavior.

### **Configuration**

TensorBoard logging is configured in your experiment config file (`configs/base.yaml` or `configs/experiments/current.yaml`):

```yaml
tensorboard:
  enabled: true                    # Enable/disable TensorBoard logging
  log_audio: true                  # Log audio samples to TensorBoard
  alpha_sweep_alphas: [0.1, 0.3, 0.5, 0.7, 0.9]  # Alpha values for linearity evaluation
  alpha_sweep_samples: 100         # Number of sample pairs for alpha sweep
```

### **Log Location**

TensorBoard logs are saved to:
```
<checkpoint_dir>/tensorboard/
```

For example, if your experiment name is `stft-vocoder`, logs will be in:
```
checkpoints/stft-vocoder/tensorboard/
```

### **Viewing Logs**

Start TensorBoard server:

```bash
# Navigate to your checkpoint directory or specify the log directory
tensorboard --logdir checkpoints/stft-vocoder/tensorboard

# Or from the project root
tensorboard --logdir checkpoints
```

Then open your browser to `http://localhost:6006` (or the URL shown in terminal).

### **Logged Metrics**

#### **Training and Validation Metrics**

All metrics are logged separately for training (`train/`) and validation (`val/`) phases:

- **Loss Components:**
  - `loss` - Total training/validation loss
  - `ReconSingle/Total`, `ReconSingle/WavL1`, `ReconSingle/MRSTFT` - Single-sample reconstruction metrics
  - `MixReconInterp/Total`, `MixReconInterp/WavL1`, `MixReconInterp/MRSTFT` - Interpolated latent reconstruction
  - `MixReconReal/Total`, `MixReconReal/WavL1`, `MixReconReal/MRSTFT` - Real mixed input reconstruction

- **Mixing Linearity Metrics:**
  - `MixRate` - Decode mixing rate (ideal = 1.0, measures how well interpolation matches real encoding)
  - `MixGap` - Difference between interpolated and real reconstructions
  - `LatentMixError` - Latent space linearity error (L2 distance)

- **Learning Rate:**
  - `lr` - Current learning rate (logged every epoch)

#### **Distribution Statistics**

For metrics like `MixRate`, distribution statistics are logged to track variability:
- `MixRate/mean` - Mean value across validation samples
- `MixRate/median` - Median value
- `MixRate/p90` - 90th percentile
- `MixRate/max` - Maximum value

#### **Audio Samples**

When `log_audio: true` is enabled, audio samples are logged when a new best model is saved:

- Original audio samples from validation set
- Reconstructed audio from the autoencoder
- These can be played directly in TensorBoard's audio tab

#### **Hyperparameters**

At training start, all hyperparameters are logged to the "HPARAMS" tab, including:
- Model architecture parameters (d_model, n_heads, n_layers, etc.)
- Training hyperparameters (batch_size, learning_rate, num_epochs, etc.)
- Loss weights (mrstft_weight, l1_weight, decode_mix_weight, etc.)
- Dataset configuration (sample_rate, chunk_seconds, etc.)

This enables easy comparison of different experiments using TensorBoard's hyperparameter comparison view.

### **Alpha Sweep Evaluation**

The system automatically runs alpha sweep evaluations at key epochs (1/3, 2/3, and final epoch) to measure linearity across different mixing coefficients. Results are logged under `AlphaSweep/` with metrics for each alpha value:

- `AlphaSweep/alpha_0.1/MixReconInterp` - Interpolated reconstruction quality at α=0.1
- `AlphaSweep/alpha_0.1/MixRate` - Decode mixing rate at α=0.1
- ... (similar for all configured alpha values)

This helps track how well the model preserves linear mixing relationships across the full range of mixing coefficients.

### **Usage Tips**

1. **Compare Experiments:** TensorBoard can load multiple log directories simultaneously:
   ```bash
   tensorboard --logdir checkpoints --port 6006
   ```
   This allows comparing different experiment runs side-by-side.

2. **Filter Metrics:** Use the regex filter in TensorBoard's scalar dashboard to focus on specific metrics (e.g., `MixRate` to see only mixing-related metrics).

3. **Monitor Training:** Watch the `val/loss` curve to identify overfitting or convergence. The `val/MixRate/mean` metric should approach 1.0 for good mixing-equivariance.

4. **Debug Audio Quality:** Check the audio tab periodically to ensure reconstructions sound reasonable. Poor audio quality in TensorBoard often correlates with high reconstruction losses.

5. **Hyperparameter Search:** Use the HPARAMS tab to compare different hyperparameter configurations and identify optimal settings.

### **Disabling TensorBoard**

To disable TensorBoard logging (e.g., to reduce I/O overhead or disk usage):

```yaml
tensorboard:
  enabled: false
```

This will skip all TensorBoard operations during training, but training will continue normally.

---

## **Status: Research Prototype**

This repository is not yet optimized for production.
Training stability, decoder design, and mixing-equivariance are active research topics.

Contributions and discussions are welcome.
