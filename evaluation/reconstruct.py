"""
Reconstruct audio from MAESTRO test split using trained autoencoder.

Key points:
- Feed stored x_stft directly; Encoder will crop frames to multiple of num_segments internally.
- Use stored x_wave as reference waveform (RMS-normalized), then optionally denormalize with rms.
"""

import os
import argparse
import random
import json

import torch
import torch.nn.functional as F
import yaml
import soundfile as sf

from models.autoencoder import Autoencoder


# -------------------------
# STFT helpers (correct shapes)
# -------------------------
def stft_ri(
    waveform: torch.Tensor,  # [L] or [B, L]
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = False,
) -> torch.Tensor:
    """
    Return real/imag tensor.
    - input:  [L]      -> output [2, F, T]
    - input:  [B, L]   -> output [B, 2, F, T]
    """
    if waveform.dim() == 1:
        x = waveform[None, :]  # [1, L]
        squeeze_batch = True
    elif waveform.dim() == 2:
        x = waveform
        squeeze_batch = False
    else:
        raise ValueError(f"waveform must be [L] or [B,L], got {waveform.shape}")

    x = x.float()
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )  # [B, F, T] complex

    out = torch.stack([X.real, X.imag], dim=1)  # [B, 2, F, T]
    return out[0] if squeeze_batch else out


def istft_ri(
    stft_ri_tensor: torch.Tensor,  # [2, F, T] or [B, 2, F, T]
    hop_length: int,
    win_length: int,
    length: int,
    center: bool = False,
    normalized: bool = False,
) -> torch.Tensor:
    """
    - input:  [2, F, T]      -> output [L]
    - input:  [B, 2, F, T]   -> output [B, L]
    """
    if stft_ri_tensor.dim() == 3:
        x = stft_ri_tensor[None, ...]  # [1,2,F,T]
        squeeze_batch = True
    elif stft_ri_tensor.dim() == 4:
        x = stft_ri_tensor
        squeeze_batch = False
    else:
        raise ValueError(f"stft_ri must be [2,F,T] or [B,2,F,T], got {stft_ri_tensor.shape}")

    x = x.float()
    real = x[:, 0]
    imag = x[:, 1]
    X = torch.complex(real, imag)  # [B,F,T]

    n_fft = (X.shape[1] - 1) * 2
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)

    wav = torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=normalized,
        length=length,
        return_complex=False,
    )  # [B,L]
    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return wav[0] if squeeze_batch else wav


def safe_peak_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    m = x.abs().max().clamp_min(eps)
    return x / m


# -------------------------
# Loading
# -------------------------
def load_model(checkpoint_path: str, device: torch.device) -> Autoencoder:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Autoencoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def load_random_chunks(chunks_dir: str, split: str, num_samples: int):
    """
    Loads random chunks from saved .pt files.
    Expects new format:
      dict with keys x_stft: [N,2,F,T], x_wave: [N,L], rms: [N]
    """
    split_dir = os.path.join(chunks_dir, split)
    index_path = os.path.join(chunks_dir, "index.json")

    with open(index_path, "r") as f:
        index = json.load(f)

    split_index = index.get(split, {})
    if not split_index:
        raise RuntimeError(f"No data for split '{split}'")

    # Build (filepath, local_idx, filename)
    all_chunks = []
    for filename, n in split_index.items():
        fp = os.path.join(split_dir, filename)
        n = int(n)
        for j in range(n):
            all_chunks.append((fp, j, filename))

    chosen = random.sample(all_chunks, k=min(num_samples, len(all_chunks)))

    # Cache per file
    cache = {}

    samples = []
    for fp, j, fname in chosen:
        if fp not in cache:
            d = torch.load(fp, map_location="cpu")  # store is safe enough for your own files
            if not (isinstance(d, dict) and "x_stft" in d and "x_wave" in d):
                raise RuntimeError(f"Unexpected format in {fp}. Need dict with x_stft and x_wave.")
            cache[fp] = d

        d = cache[fp]
        samples.append(
            {
                "x_stft": d["x_stft"][j],   # [2,F,T]
                "x_wave": d["x_wave"][j],   # [L] RMS-normalized
                "rms": (d["rms"][j].item() if "rms" in d and d["rms"] is not None else None),
                "source": fname,
                "chunk_idx": j,
            }
        )
    return samples


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--peak_norm", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model + read important params from checkpoint config
    ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_config"]
    num_segments = int(model_cfg["num_segments"])
    target_length = int(model_cfg["target_length"])

    model = load_model(cfg["checkpoint"], device)

    # STFT params for analysis / cheat-phase audio
    stft_cfg = cfg["stft"]
    n_fft = int(stft_cfg.get("n_fft", 1024))
    hop_length = int(stft_cfg["hop_length"])
    win_length = int(stft_cfg["win_length"])
    stft_center = bool(model_cfg.get("stft_center", False))  # should match training MRSTFT choice

    sample_rate = int(cfg["data"]["sample_rate"])
    chunk_seconds = float(cfg["data"]["chunk_seconds"])
    chunk_samples = int(sample_rate * chunk_seconds)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Load samples
    samples = load_random_chunks(
        chunks_dir=cfg["data"]["chunks_dir"],
        split=cfg["data"]["split"],
        num_samples=int(cfg["num_samples"]),
    )

    print(f"Loaded {len(samples)} samples from split={cfg['data']['split']}.")
    print(f"Model num_segments={num_segments}, target_length={target_length}, chunk_samples={chunk_samples}")
    if target_length != chunk_samples:
        print("Warning: target_length != chunk_samples. Reconstruction will be trimmed/padded to chunk_samples for saving.")

    for i, s in enumerate(samples):
        x_stft = s["x_stft"].float()              # [2,F,T_full]
        x_wave = s["x_wave"].float()              # [L] RMS-normalized
        rms = s.get("rms", None)

        # Ensure waveform length matches chunk_samples (for saving / comparison)
        if x_wave.numel() > chunk_samples:
            x_wave = x_wave[:chunk_samples]
        elif x_wave.numel() < chunk_samples:
            x_wave = F.pad(x_wave, (0, chunk_samples - x_wave.numel()), value=0.0)

        # Model inference: feed STFT as trained (encoder will crop internally to multiple of num_segments)
        with torch.no_grad():
            z = model.encoder(x_stft[None, ...].to(device))       # [1,S,D]
            y_hat = model.decoder(z)[0, 0].detach().cpu().float() # [target_length]

        # Trim/pad recon to chunk_samples for saving
        if y_hat.numel() > chunk_samples:
            y_hat = y_hat[:chunk_samples]
        elif y_hat.numel() < chunk_samples:
            y_hat = F.pad(y_hat, (0, chunk_samples - y_hat.numel()), value=0.0)

        # Denormalize RMS for listening (optional but typically what you want)
        audio_orig = x_wave
        audio_recon = y_hat
        if rms is not None:
            audio_orig = audio_orig * rms
            audio_recon = audio_recon * rms

        if args.peak_norm:
            audio_orig = safe_peak_norm(audio_orig)
            audio_recon = safe_peak_norm(audio_recon)

        print(
            f"[{i+1}/{len(samples)}] {s['source']} chunk={s['chunk_idx']} "
            f"peak(orig)={audio_orig.abs().max().item():.4f} "
            f"peak(recon)={audio_recon.abs().max().item():.4f} "
            f"{'rms='+format(rms, '.4f') if rms is not None else ''}"
        )

        orig_path = os.path.join(output_dir, f"sample_{i:03d}_original.wav")
        recon_path = os.path.join(output_dir, f"sample_{i:03d}_reconstructed.wav")

        sf.write(orig_path, audio_orig.numpy(), sample_rate, subtype="PCM_16")
        sf.write(recon_path, audio_recon.numpy(), sample_rate, subtype="PCM_16")

    print(f"Saved {len(samples)} pairs to: {output_dir}")


if __name__ == "__main__":
    main()
