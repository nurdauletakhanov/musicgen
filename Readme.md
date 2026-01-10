# Mixing-Equivariant Waveform Tokenizer

A research prototype for learning 1-second audio tokens whose latent space preserves **linear mixing** relationships.

This project aims to develop a **waveform-based audio tokenizer** that compresses a 1-second waveform into a small latent representation (continuous or quantized), while enforcing the property:

[
\text{emb}(\lambda x_1 + (1-\lambda),x_2);\approx;
\lambda,\text{emb}(x_1) + (1-\lambda),\text{emb}(x_2)
]

This *mixing-equivariance constraint* is hypothesized to improve controllability, interpolation, and downstream generative modeling by aligning the latent space with the linear structure of the waveform domain (which is physically linear for sound pressure).

---

## **Project Goals**

### 1. Build a tokenizer for 1-second audio chunks

* Pure waveform input (22k–24k Hz)
* CNN segment encoder → Transformer → global token

### 2. Enforce **mixing linearity**

Given audio chunks `x1`, `x2`, and mixture `xλ = λ x1 + (1−λ)x2`, the tokenizer should satisfy:

* **Latent linearity**
  [
  f(x_\lambda) \approx \lambda f(x_1) + (1 - \lambda) f(x_2)
  ]

* **Reconstruction linearity** (optional)
  [
  g(f(x_\lambda)) \approx \lambda g(f(x_1)) + (1-\lambda)g(f(x_2))
  ]

### 3. Provide a clean benchmark for comparing tokenizers

* Reconstruction metrics: MR-STFT, log-magnitude loss, L1 waveform
* Latent space metrics: linearity error, PCA analysis, λ-prediction error
* Ablations: segment lengths, d_model sizes, RVQ depth

---

## **Architecture Overview**

### **1. Segmentation**

Each 1-second waveform chunk `x ∈ ℝ^{1×L}` is divided into **T segments**:

```
L = sample_rate * 1.0  
T = num_tokens   # e.g., 40  
segment_len = L // T
```

### **2. Local encoder (CNN over segments)**

Each segment (≈551 samples at 22kHz with T=40) is encoded into a `d_model`-dimensional token:

```
[B, T, 1, segment_len] → [B, T, d_model]
```

This uses a lightweight Conv1d stack followed by AdaptiveAvgPool.

### **3. Global encoder (Transformer)**

A Transformer with a learnable `[CLS]` token models the sequence of segment embeddings:

```
input:  [CLS, token1, token2, ... token_T]
output: CLS embedding z ∈ ℝ^{d_model}
```

This is the **1-second latent token**.

### **4. Decoder**

Maps the latent token back into a waveform:

```
z → ConvTranspose stack → waveform of length L
```

A future version may use a diffusion decoder.

## **Loss Functions**

### **1. Reconstruction Loss**

Combination of MR-STFT + L1 waveform:

[
\mathcal{L}_\text{recon}(x,\hat{x}) = \alpha;\text{MRSTFT}(x,\hat{x}) + \beta;|x - \hat{x}|_1
]

### **2. Latent Mixing Loss**

For randomly sampled `(x1, x2)` and λ∼Uniform(0,1):

[
x_\lambda = \lambda x_1 + (1-\lambda)x_2
]

[
\mathcal{L}*\text{mix-latent} = \big|f(x*\lambda) - [\lambda f(x_1) + (1-\lambda)f(x_2)]\big|_2^2
]

### **3. Reconstruction Mixing Loss (optional)**

[
\mathcal{L}*\text{mix-wave} = \text{MRSTFT}\big(x*\lambda,;\lambda\hat{x}_1+(1-\lambda)\hat{x}_2\big)
]

### **Total Loss**

[
\mathcal{L} = \mathcal{L}*\text{recon} + \gamma\mathcal{L}*\text{mix-latent} + \delta\mathcal{L}_\text{mix-wave}
]

---

## **Why This Project is Interesting**

### **Most existing tokenizers:**

* Work with 10–40 ms windows (Encodec, DAC, SoundStream).
* Produce long sequences of tokens.
* Are not designed to preserve the linear structure of the waveform.
* Are optimized mainly for perceptual quality and compression.

### **This project attempts something new:**

1. **A 1-second representation** (much more compressed).
2. **Mixing-equivariance as a core constraint**.
3. **Direct waveform modeling**, not STFT or mel-spectrogram.
4. **Compatibility with simple latent generative models** (AR, diffusion).
5. **New metrics** for evaluating linearity in audio embeddings.

This could open new avenues for:

* Audio interpolation and morphing
* Style transfer
* Layered music creation (blending stems)
* Generative models that obey physical mixing rules
* Tokenizers for downstream music models (MusicGen-style)

---

## **Repository Structure**

Expected layout:

```
project/
│
├── encoder/
│   ├── segment_cnn.py
│   ├── global_transformer.py
│   └── encoder.py
│
├── decoder/
│   └── decoder.py
│
├── training/
│   ├── dataset.py
│   ├── mixing_loss.py
│   ├── train.py
│   └── utils.py
│
├── experiments/
│   ├── baseline_recon.ipynb
│   ├── mixing_equivariance.ipynb
│   └── rvq_ablation.ipynb
│
├── README.md
└── requirements.txt
```

---

## **Example Usage**

### **Forward pass**

```python
encoder = Encoder(d_model=256, num_tokens=40)
decoder = WaveformDecoder(d_model=256, target_length=22050)

x = torch.randn(8, 1, 22050)  # batch of 1-second clips

z = encoder(x)
x_hat = decoder(z)
```

### **Latent mixing**

```python
z_mix = λ * z1 + (1 - λ) * z2
x_mix = decoder(z_mix)
```

---

## **Planned Research Experiments**

### **Phase A: Baseline**

* Train the AE without mixing loss.
* Evaluate reconstruction and latent PCA.

### **Phase B: Add mixing-equivariant loss**

* Measure linearity error across λ ∈ [0,1].
* Test generalization to unseen mixtures.

### **Phase C: Add RVQ**

* Measure the tradeoff: reconstruction vs. linearity vs. bitrate.

### **Phase D: Generation**

* Train a small AR or diffusion model over 1-second tokens.
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
# In evaluation/reconstruct.py or evaluation/sample_diffusion.py
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
Training stability, decoder design, and quantization are active research topics.

Contributions and discussions are welcome.
