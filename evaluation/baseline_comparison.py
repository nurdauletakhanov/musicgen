"""
Baseline comparison: Evaluate Encodec and DAC on MixRate.

Tests whether existing neural audio codecs preserve linear mixing in their
latent spaces. Uses continuous (pre-quantization) embeddings for fair comparison.

Usage:
    python -m evaluation.baseline_comparison \
        --chunks-dir ./musdb-chunks-stft \
        --output results/baseline_comparison.json

Dependencies:
    pip install encodec descript-audio-codec
"""

import argparse
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def load_test_waveforms(chunks_dir: str, num_samples: int = 200) -> List[Dict]:
    """Load stem pair waveforms from MUSDB test split."""
    index_path = os.path.join(chunks_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    test_tracks = index.get("test", {})
    if not test_tracks:
        raise RuntimeError("No test data found in index.json")

    stem_order = ["drums", "bass", "other", "vocals"]
    stem_pairs = [
        ("drums", "vocals"),
        ("drums", "bass"),
        ("bass", "vocals"),
        ("other", "vocals"),
        ("drums", "other"),
        ("bass", "other"),
    ]

    samples = []
    for track_key, track_info in test_tracks.items():
        if not isinstance(track_info, dict) or "stems" not in track_info:
            continue
        stems = track_info["stems"]
        n_chunks = track_info.get("num_chunks", 0)
        if not all(s in stems for s in stem_order):
            continue

        stem_data = {}
        for stem_name in stem_order:
            path = os.path.join(chunks_dir, "test", stems[stem_name])
            if not os.path.exists(path):
                continue
            data = torch.load(path, map_location="cpu", weights_only=True)
            stem_data[stem_name] = data["x_wave"].float()

        if len(stem_data) < 4:
            continue

        for chunk_idx in range(min(n_chunks, 5)):
            for stem1_name, stem2_name in stem_pairs:
                if stem1_name not in stem_data or stem2_name not in stem_data:
                    continue
                samples.append({
                    "x1_wave": stem_data[stem1_name][chunk_idx],
                    "x2_wave": stem_data[stem2_name][chunk_idx],
                    "track": track_key,
                    "stems": f"{stem1_name}+{stem2_name}",
                })

    random.seed(42)
    random.shuffle(samples)
    return samples[:num_samples]


def compute_mrstft_loss(x_hat, x, device, eps=1e-8):
    """Compute multi-resolution STFT loss (same as our model)."""
    ffts = [512, 1024, 2048]
    hops = [256, 512, 128]
    wins = [512, 1024, 2048]

    total_sc = 0.0
    total_mag = 0.0

    for n_fft, hop, win in zip(ffts, hops, wins):
        if x_hat.dim() == 3:
            x_hat_s = x_hat.squeeze(1)
        else:
            x_hat_s = x_hat
        if x.dim() == 3:
            x_s = x.squeeze(1)
        else:
            x_s = x

        window = torch.hann_window(win, device=device)
        M = torch.stft(x_s.float(), n_fft=n_fft, hop_length=hop, win_length=win,
                        window=window, center=False, return_complex=True).abs().clamp(min=eps)
        Mhat = torch.stft(x_hat_s.float(), n_fft=n_fft, hop_length=hop, win_length=win,
                          window=window, center=False, return_complex=True).abs().clamp(min=eps)

        num = torch.linalg.norm(M - Mhat, dim=(1, 2))
        den = torch.linalg.norm(M, dim=(1, 2)).clamp(min=eps)
        sc = (num / den).mean()
        mag = F.l1_loss(torch.log(Mhat + eps), torch.log(M + eps))

        total_sc += sc
        total_mag += mag

    return (total_sc + total_mag) / len(ffts)


def resample_if_needed(wave, orig_sr, target_sr):
    """Resample waveform if sample rates differ."""
    if orig_sr == target_sr:
        return wave
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr).to(wave.device)
    return resampler(wave)


# ---- Encodec evaluation ----

def evaluate_encodec(samples, alphas, device, num_samples=200):
    """Evaluate Encodec's latent space linearity using continuous embeddings."""
    try:
        from encodec import EncodecModel
    except ImportError:
        print("  encodec not installed. Run: pip install encodec")
        return None

    print("  Loading Encodec 48kHz model...")
    model = EncodecModel.encodec_model_48khz()
    model.to(device).eval()
    model.set_target_bandwidth(6.0)

    our_sr = 44100
    enc_sr = 48000

    results = {}
    for alpha in alphas:
        beta = 1.0 - alpha
        rates = []

        for sample in samples[:num_samples]:
            x1 = sample["x1_wave"].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, L]
            x2 = sample["x2_wave"].unsqueeze(0).unsqueeze(0).to(device)

            # Resample to 48kHz for Encodec
            x1_48 = resample_if_needed(x1.squeeze(0), our_sr, enc_sr).unsqueeze(0)
            x2_48 = resample_if_needed(x2.squeeze(0), our_sr, enc_sr).unsqueeze(0)

            # Encodec 48kHz expects stereo (2 channels) — duplicate mono to stereo
            if model.channels == 2 and x1_48.shape[1] == 1:
                x1_48 = x1_48.expand(-1, 2, -1)
                x2_48 = x2_48.expand(-1, 2, -1)

            # Mix at 48kHz
            x_mix_48 = alpha * x1_48 + beta * x2_48

            with torch.no_grad():
                # Continuous embeddings (pre-quantization)
                z1 = model.encoder(x1_48)
                z2 = model.encoder(x2_48)
                z_real = model.encoder(x_mix_48)

                # Oracle: decode real mix embedding
                x_real_recon = model.decoder(z_real)

                # Latent interpolation
                z_interp = alpha * z1 + beta * z2
                x_interp = model.decoder(z_interp)

            # Convert stereo to mono if needed, then resample back to 44.1kHz
            if x_mix_48.shape[1] == 2:
                x_mix_48_mono = x_mix_48.mean(dim=1, keepdim=True)
                x_real_mono = x_real_recon.mean(dim=1, keepdim=True)
                x_interp_mono = x_interp.mean(dim=1, keepdim=True)
            else:
                x_mix_48_mono = x_mix_48
                x_real_mono = x_real_recon
                x_interp_mono = x_interp

            x_mix_44 = resample_if_needed(x_mix_48_mono.squeeze(0), enc_sr, our_sr).unsqueeze(0)
            x_real_44 = resample_if_needed(x_real_mono.squeeze(0), enc_sr, our_sr).unsqueeze(0)
            x_interp_44 = resample_if_needed(x_interp_mono.squeeze(0), enc_sr, our_sr).unsqueeze(0)

            # Trim to same length
            min_len = min(x_mix_44.shape[-1], x_real_44.shape[-1], x_interp_44.shape[-1])
            x_mix_44 = x_mix_44[..., :min_len]
            x_real_44 = x_real_44[..., :min_len]
            x_interp_44 = x_interp_44[..., :min_len]

            # Compute losses
            l1_real = F.l1_loss(x_real_44, x_mix_44).item()
            mr_real = compute_mrstft_loss(x_real_44, x_mix_44, device).item()
            l1_interp = F.l1_loss(x_interp_44, x_mix_44).item()
            mr_interp = compute_mrstft_loss(x_interp_44, x_mix_44, device).item()

            loss_real = l1_real + mr_real
            loss_interp = l1_interp + mr_interp
            rate = loss_interp / (loss_real + 1e-8)
            rates.append(rate)

        results[alpha] = {
            "MixRate_mean": float(np.mean(rates)),
            "MixRate_std": float(np.std(rates)),
            "MixRate_median": float(np.median(rates)),
            "MixRate_p90": float(np.percentile(rates, 90)),
            "num_samples": len(rates),
        }
        print(f"    alpha={alpha}: MixRate={np.mean(rates):.4f} ± {np.std(rates):.4f}")

    return results


# ---- DAC evaluation ----

def evaluate_dac(samples, alphas, device, num_samples=200):
    """Evaluate DAC's latent space linearity using continuous embeddings."""
    try:
        import dac
        from dac.utils import download
    except ImportError:
        print("  dac not installed. Run: pip install descript-audio-codec")
        return None

    print("  Loading DAC 44kHz model...")
    model_path = download(model_type="44khz")
    model = dac.DAC.load(model_path)
    model.to(device).eval()

    our_sr = 44100

    results = {}
    for alpha in alphas:
        beta = 1.0 - alpha
        rates = []

        for sample in samples[:num_samples]:
            x1 = sample["x1_wave"].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, L]
            x2 = sample["x2_wave"].unsqueeze(0).unsqueeze(0).to(device)
            x_mix = alpha * x1 + beta * x2

            with torch.no_grad():
                # DAC encoder returns continuous embeddings
                z1, _, _, _, _ = model.encode(x1)
                z2, _, _, _, _ = model.encode(x2)
                z_real, _, _, _, _ = model.encode(x_mix)

                # Oracle
                x_real_recon = model.decode(z_real)

                # Latent interpolation
                z_interp = alpha * z1 + beta * z2
                x_interp = model.decode(z_interp)

            # Trim to same length
            min_len = min(x_mix.shape[-1], x_real_recon.shape[-1], x_interp.shape[-1])
            x_mix_t = x_mix[..., :min_len]
            x_real_t = x_real_recon[..., :min_len]
            x_interp_t = x_interp[..., :min_len]

            l1_real = F.l1_loss(x_real_t, x_mix_t).item()
            mr_real = compute_mrstft_loss(x_real_t, x_mix_t, device).item()
            l1_interp = F.l1_loss(x_interp_t, x_mix_t).item()
            mr_interp = compute_mrstft_loss(x_interp_t, x_mix_t, device).item()

            loss_real = l1_real + mr_real
            loss_interp = l1_interp + mr_interp
            rate = loss_interp / (loss_real + 1e-8)
            rates.append(rate)

        results[alpha] = {
            "MixRate_mean": float(np.mean(rates)),
            "MixRate_std": float(np.std(rates)),
            "MixRate_median": float(np.median(rates)),
            "MixRate_p90": float(np.percentile(rates, 90)),
            "num_samples": len(rates),
        }
        print(f"    alpha={alpha}: MixRate={np.mean(rates):.4f} ± {np.std(rates):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Encodec and DAC baseline MixRate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chunks-dir", type=str, default="./musdb-chunks-stft")
    parser.add_argument("--output", type=str, default="results/baseline_comparison.json")
    parser.add_argument("--num-samples", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("Loading test stem pairs...")
    samples = load_test_waveforms(args.chunks_dir, args.num_samples)
    print(f"Loaded {len(samples)} test stem pairs\n")

    all_results = {}

    # Encodec
    print("Evaluating Encodec 48kHz...")
    encodec_results = evaluate_encodec(samples, alphas, device, args.num_samples)
    if encodec_results:
        all_results["encodec_48khz"] = encodec_results
        mean_rates = [encodec_results[a]["MixRate_mean"] for a in alphas]
        print(f"  Encodec overall MixRate: {np.mean(mean_rates):.4f}\n")

    # DAC
    print("Evaluating DAC 44kHz...")
    dac_results = evaluate_dac(samples, alphas, device, args.num_samples)
    if dac_results:
        all_results["dac_44khz"] = dac_results
        mean_rates = [dac_results[a]["MixRate_mean"] for a in alphas]
        print(f"  DAC overall MixRate: {np.mean(mean_rates):.4f}\n")

    # Summary
    print("=" * 60)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'alpha=0.1':>10} {'0.3':>10} {'0.5':>10} {'0.7':>10} {'0.9':>10} {'Mean':>10}")
    print("-" * 80)
    for model_name, results in all_results.items():
        rates = [results[a]["MixRate_mean"] for a in alphas]
        row = f"{model_name:<20}"
        for r in rates:
            row += f" {r:>9.4f}"
        row += f" {np.mean(rates):>9.4f}"
        print(row)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
