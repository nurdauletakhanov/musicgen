"""
Reconstruct audio from MAESTRO test split using trained autoencoder.

Usage:
    # Basic usage with checkpoint path
    python evaluation/reconstruct.py --checkpoint checkpoints/my-experiment/best_model.pth

    # With custom options
    python evaluation/reconstruct.py --checkpoint checkpoints/my-experiment/best_model.pth \
        --num-samples 20 --output-dir ./my-reconstructions --peak-norm

    # Using HuggingFace Hub checkpoint
    python evaluation/reconstruct.py --checkpoint username/model-name

Key points:
- Feed stored x_stft directly; Encoder will crop frames to multiple of num_segments internally.
- Use stored x_wave as reference waveform (RMS-normalized), then optionally denormalize with rms.
- STFT and data parameters are read from the checkpoint's model_config when possible.
"""

import argparse
import json
import os
import random

import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

from models.autoencoder import Autoencoder
from training.hub_utils import resolve_checkpoint_path
from utils.audio import safe_peak_norm


def load_model(checkpoint_path: str, device: torch.device) -> Autoencoder:
    """Load model from checkpoint."""
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    if resolved_path != checkpoint_path:
        print(f"Downloaded checkpoint from Hub to: {resolved_path}")
    
    ckpt = torch.load(resolved_path, map_location=device, weights_only=False)
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
            d = torch.load(fp, map_location="cpu")
            if not (isinstance(d, dict) and "x_stft" in d and "x_wave" in d):
                raise RuntimeError(f"Unexpected format in {fp}. Need dict with x_stft and x_wave.")
            cache[fp] = d

        d = cache[fp]
        samples.append(
            {
                "x_stft": d["x_stft"][j],
                "x_wave": d["x_wave"][j],
                "rms": (d["rms"][j].item() if "rms" in d and d["rms"] is not None else None),
                "source": fname,
                "chunk_idx": j,
            }
        )
    return samples


def try_load_checkpoint_config(checkpoint_path: str) -> dict:
    """
    Try to load config.yaml from the same directory as the checkpoint.
    Returns empty dict if not found.
    """
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint_dir = os.path.dirname(resolved_path)
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct audio samples using trained autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to checkpoint file or HuggingFace Hub ID"
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="./maestro-chunks-stft",
        help="Directory containing preprocessed STFT chunks"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to use (train/validation/test)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to reconstruct"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation/reconstructions",
        help="Output directory for reconstructed audio"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Audio sample rate"
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=1.0,
        help="Duration of each chunk in seconds"
    )
    parser.add_argument(
        "--peak-norm",
        action="store_true",
        help="Apply peak normalization to output audio"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve checkpoint path (download from Hub if necessary)
    resolved_checkpoint = resolve_checkpoint_path(args.checkpoint)
    if resolved_checkpoint != args.checkpoint:
        print(f"Downloaded checkpoint from Hub to: {resolved_checkpoint}")
    
    # Try to load config from checkpoint directory
    checkpoint_config = try_load_checkpoint_config(args.checkpoint)
    
    # Use config values as defaults if available, CLI args take precedence
    chunks_dir = args.chunks_dir
    if chunks_dir == "./maestro-chunks-stft" and checkpoint_config.get("data", {}).get("chunks_dir"):
        chunks_dir = checkpoint_config["data"]["chunks_dir"]
    
    sample_rate = args.sample_rate
    if args.sample_rate == 44100 and checkpoint_config.get("data", {}).get("sample_rate"):
        sample_rate = checkpoint_config["data"]["sample_rate"]
    
    chunk_seconds = args.chunk_seconds
    if args.chunk_seconds == 1.0 and checkpoint_config.get("data", {}).get("chunk_seconds"):
        chunk_seconds = checkpoint_config["data"]["chunk_seconds"]
    
    # Load model + read important params from checkpoint config
    ckpt = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_config"]
    num_segments = int(model_cfg["num_segments"])
    target_length = int(model_cfg["target_length"])

    model = load_model(resolved_checkpoint, device)

    chunk_samples = int(sample_rate * chunk_seconds)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load samples
    samples = load_random_chunks(
        chunks_dir=chunks_dir,
        split=args.split,
        num_samples=args.num_samples,
    )

    print(f"Loaded {len(samples)} samples from split={args.split}.")
    print(f"Model num_segments={num_segments}, target_length={target_length}, chunk_samples={chunk_samples}")
    if target_length != chunk_samples:
        print("Warning: target_length != chunk_samples. Reconstruction will be trimmed/padded.")

    for i, s in enumerate(samples):
        x_stft = s["x_stft"].float()
        x_wave = s["x_wave"].float()
        rms = s.get("rms", None)

        # Ensure waveform length matches chunk_samples
        if x_wave.numel() > chunk_samples:
            x_wave = x_wave[:chunk_samples]
        elif x_wave.numel() < chunk_samples:
            x_wave = F.pad(x_wave, (0, chunk_samples - x_wave.numel()), value=0.0)

        # Model inference
        with torch.no_grad():
            z = model.encoder(x_stft[None, ...].to(device))
            y_hat = model.decoder(z)[0, 0].detach().cpu().float()

        # Trim/pad recon to chunk_samples for saving
        if y_hat.numel() > chunk_samples:
            y_hat = y_hat[:chunk_samples]
        elif y_hat.numel() < chunk_samples:
            y_hat = F.pad(y_hat, (0, chunk_samples - y_hat.numel()), value=0.0)

        # Denormalize RMS for listening
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
