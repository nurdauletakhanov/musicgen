"""
Generate spectrogram comparison figures for NeurIPS paper.

Shows ground truth mix vs oracle D(E(A+B)) vs latent mix D(E(A)+E(B))
with per-panel reconstruction loss annotations.

Usage:
    python -m evaluation.plot_spectrograms \
        --checkpoint checkpoints/musdb-phase2-mixing/checkpoint_<epoch>.pth \
        --output paper/figures/
"""

import argparse
import json
import os
import random
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import torch.nn.functional as F

from models.autoencoder import Autoencoder

matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_model(checkpoint_path: str, device: torch.device) -> Autoencoder:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Autoencoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def load_stem_pairs(chunks_dir: str, num_samples: int = 5) -> List[Dict]:
    index_path = os.path.join(chunks_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    test_tracks = index.get("test", {})
    stem_order = ["drums", "bass", "other", "vocals"]
    stem_pairs_list = [("drums", "vocals"), ("bass", "other"), ("drums", "bass")]

    samples = []
    for track_key, track_info in test_tracks.items():
        if not isinstance(track_info, dict) or "stems" not in track_info:
            continue
        stems = track_info["stems"]
        if not all(s in stems for s in stem_order):
            continue

        stem_data = {}
        for stem_name in stem_order:
            path = os.path.join(chunks_dir, "test", stems[stem_name])
            if not os.path.exists(path):
                continue
            data = torch.load(path, map_location="cpu", weights_only=True)
            stem_data[stem_name] = {
                "x_stft": data["x_stft"].float(),
                "x_wave": data["x_wave"].float(),
            }

        if len(stem_data) < 4:
            continue

        for s1, s2 in stem_pairs_list:
            samples.append({
                "x1_stft": stem_data[s1]["x_stft"][0],
                "x1_wave": stem_data[s1]["x_wave"][0],
                "x2_stft": stem_data[s2]["x_stft"][0],
                "x2_wave": stem_data[s2]["x_wave"][0],
                "track": track_key,
                "stems": f"{s1}+{s2}",
            })

    random.seed(42)
    random.shuffle(samples)
    return samples[:num_samples]


def wave_to_spectrogram(wave, n_fft=2048, hop=512):
    """Convert waveform tensor to log-magnitude spectrogram [F, T]."""
    if wave.dim() == 3:
        wave = wave.squeeze(0).squeeze(0)
    elif wave.dim() == 2:
        wave = wave.squeeze(0)
    wave = wave.cpu().float()
    window = torch.hann_window(n_fft)
    stft = torch.stft(wave, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=window, center=True, return_complex=True)
    mag = stft.abs()  # [F, T]
    return 20 * torch.log10(mag + 1e-8).numpy()


def plot_spectrogram_comparison(
    model, sample, device, output_path, sr=44100
):
    """Generate a 3-panel spectrogram comparison with loss annotations."""
    x1_stft = sample["x1_stft"].unsqueeze(0).to(device)
    x1_wave = sample["x1_wave"].unsqueeze(0).to(device)
    x2_stft = sample["x2_stft"].unsqueeze(0).to(device)
    x2_wave = sample["x2_wave"].unsqueeze(0).to(device)

    if x1_wave.dim() == 2:
        x1_wave = x1_wave.unsqueeze(1)
        x2_wave = x2_wave.unsqueeze(1)

    alpha = 0.5

    # Run inference in float32 (no autocast) to avoid half-precision artifacts
    with torch.no_grad():
        z1 = model.encoder(x1_stft)
        z2 = model.encoder(x2_stft)

        x_mix_wave = alpha * x1_wave + (1 - alpha) * x2_wave
        x_mix_stft = model._compute_stft_from_wave(x_mix_wave)
        z_real = model.encoder(x_mix_stft)
        x_oracle = model.decoder(z_real)

        z_interp = alpha * z1 + (1 - alpha) * z2
        x_latent = model.decoder(z_interp)

        # Compute losses for annotation
        l1_oracle = F.l1_loss(x_oracle, x_mix_wave).item()
        l1_latent = F.l1_loss(x_latent, x_mix_wave).item()

    # Build panels: title, spectrogram data, annotation
    panels = [
        ("Ground Truth Mix", wave_to_spectrogram(x_mix_wave), None),
        (r"Oracle: $D(E(A\!+\!B))$", wave_to_spectrogram(x_oracle), f"L1 = {l1_oracle:.4f}"),
        (r"Latent Mix: $D(E(A)\!+\!E(B))$", wave_to_spectrogram(x_latent), f"L1 = {l1_latent:.4f}"),
    ]

    # Create figure with GridSpec for proper colorbar placement
    fig = plt.figure(figsize=(15, 3.5))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.04], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[0, 3])

    all_specs = [p[1] for p in panels]
    vmax = max(s.max() for s in all_specs)
    vmin = max(min(s.min() for s in all_specs), vmax - 80)

    duration = x_mix_wave.shape[-1] / sr
    freq_max = sr / 2 / 1000  # kHz
    # Limit display to 16 kHz for better visibility
    display_freq_max = 16.0
    n_fft = 2048
    freq_bin_max = int(display_freq_max / freq_max * (n_fft // 2 + 1))

    for ax, (title, spec, annotation) in zip(axes, panels):
        # Crop to display_freq_max
        spec_cropped = spec[:freq_bin_max, :]
        im = ax.imshow(spec_cropped, aspect='auto', origin='lower', vmin=vmin, vmax=vmax,
                       cmap='magma', extent=[0, duration, 0, display_freq_max])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Time (s)')

        if annotation:
            ax.text(0.02, 0.95, annotation, transform=ax.transAxes,
                    fontsize=10, color='white', fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

    axes[0].set_ylabel('Frequency (kHz)')

    fig.colorbar(im, cax=cbar_ax, label='Magnitude (dB)')

    fig.suptitle(f'{sample["track"][:35]} — {sample["stems"]}', fontsize=11, y=1.01)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved spectrogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spectrogram comparison figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--chunks-dir", type=str, default="./musdb-chunks-stft")
    parser.add_argument("--output", type=str, default="results/figures")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)

    print(f"Loading stem pairs from {args.chunks_dir}...")
    samples = load_stem_pairs(args.chunks_dir, args.num_examples)
    print(f"Loaded {len(samples)} samples")

    os.makedirs(args.output, exist_ok=True)

    for i, sample in enumerate(samples):
        output_path = os.path.join(args.output, f"spectrogram_{i:02d}.{args.format}")
        plot_spectrogram_comparison(model, sample, device, output_path)

    print(f"\nAll spectrograms saved to {args.output}/")


if __name__ == "__main__":
    main()
