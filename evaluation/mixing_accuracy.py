"""
Mixing accuracy evaluation using MUSDB18 stem pairs.

Two evaluation modes:

1. Stem-pair alpha sweep: Like alpha_sweep but using real stem pairs
   instead of random dataset pairs. More meaningful since stems are
   genuine additive components of a mix.

2. Full-stem summation test: Encode all 4 stems, sum latents, decode,
   compare against the actual mixture. Tests if D(z1+z2+z3+z4) ≈ mixture.
"""

import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast

from data.dataloader_musdb import MusdbStemPairDataset


def run_stem_alpha_sweep(
    model: torch.nn.Module,
    dataset: MusdbStemPairDataset,
    alphas: List[float],
    num_samples: int = 100,
    device: torch.device = None,
    use_amp: bool = True,
) -> Dict[float, Dict[str, float]]:
    """
    Alpha sweep using real stem pairs from MUSDB18.

    Same metrics as alpha_sweep.run_alpha_sweep but pairs come from
    aligned stems (e.g., drums + vocals from the same time position).

    Args:
        model: Autoencoder model
        dataset: MusdbStemPairDataset (returns x_stft, x_wave, x_stft2, x_wave2)
        alphas: Fixed alpha values to evaluate
        num_samples: Number of stem pairs to use
        device: Inference device

    Returns:
        Dict mapping alpha -> metrics dict
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    num_samples = min(num_samples, len(dataset))

    # Deterministic indices
    random.seed(42)
    indices = random.sample(range(len(dataset)), num_samples)

    results = {}
    with torch.no_grad():
        for alpha in alphas:
            results[alpha] = _evaluate_stem_alpha(
                model, dataset, indices, alpha, device, use_amp
            )

    return results


def _evaluate_stem_alpha(
    model: torch.nn.Module,
    dataset: MusdbStemPairDataset,
    indices: List[int],
    alpha: float,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Evaluate a single alpha on stem pairs."""
    beta = 1.0 - alpha

    interp_l1_sum = 0.0
    interp_mr_sum = 0.0
    real_l1_sum = 0.0
    real_mr_sum = 0.0
    count = 0

    for idx in indices:
        sample = dataset[idx]

        x1_stft = sample['x_stft'].unsqueeze(0).to(device)
        x1_wave = sample['x_wave'].unsqueeze(0).to(device)
        x2_stft = sample['x_stft2'].unsqueeze(0).to(device)
        x2_wave = sample['x_wave2'].unsqueeze(0).to(device)

        if x1_wave.dim() == 2:
            x1_wave = x1_wave.unsqueeze(1)
            x2_wave = x2_wave.unsqueeze(1)

        # Truncate to decoder target_length (handles padded preprocessing)
        tgt = model.decoder.target_length
        if x1_wave.size(-1) > tgt:
            x1_wave = x1_wave[..., :tgt]
            x2_wave = x2_wave[..., :tgt]

        with autocast("cuda", enabled=use_amp):
            z1 = model.encoder(x1_stft)
            z2 = model.encoder(x2_stft)

            x_mix_wave = alpha * x1_wave + beta * x2_wave

            x_mix_stft = model._compute_stft_from_wave(x_mix_wave)
            z_real = model.encoder(x_mix_stft)
            x_real_recon = model.decoder(z_real)

            z_interp = alpha * z1 + beta * z2
            x_interp = model.decoder(z_interp)

            l1_real = F.l1_loss(x_real_recon, x_mix_wave).item()
            mr_real = model.mrstft_loss(x_real_recon, x_mix_wave).item()
            l1_interp = F.l1_loss(x_interp, x_mix_wave).item()
            mr_interp = model.mrstft_loss(x_interp, x_mix_wave).item()

        interp_l1_sum += l1_interp
        interp_mr_sum += mr_interp
        real_l1_sum += l1_real
        real_mr_sum += mr_real
        count += 1

    interp_l1 = interp_l1_sum / count
    interp_mr = interp_mr_sum / count
    real_l1 = real_l1_sum / count
    real_mr = real_mr_sum / count

    mix_l1_weight = getattr(model, 'mix_l1_weight', 1.0)
    mix_mrstft_weight = getattr(model, 'mix_mrstft_weight', 1.0)

    interp_total = mix_l1_weight * interp_l1 + mix_mrstft_weight * interp_mr
    real_total = mix_l1_weight * real_l1 + mix_mrstft_weight * real_mr

    rate = interp_total / (real_total + 1e-8)
    gap = interp_total - real_total

    return {
        'MixReconInterp': interp_total,
        'MixReconInterp/WavL1': interp_l1,
        'MixReconInterp/MRSTFT': interp_mr,
        'MixReconReal': real_total,
        'MixReconReal/WavL1': real_l1,
        'MixReconReal/MRSTFT': real_mr,
        'MixRate': rate,
        'MixGap': gap,
    }


def run_full_stem_summation(
    model: torch.nn.Module,
    chunks_dir: str,
    num_tracks: int = 10,
    chunks_per_track: int = 10,
    device: torch.device = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Full-stem summation test: encode all 4 stems, sum latents, decode,
    compare against the real mixture waveform.

    Tests: D(z_drums + z_bass + z_other + z_vocals) ≈ drums + bass + other + vocals

    Args:
        model: Autoencoder model
        chunks_dir: Path to musdb-chunks-stft/ directory
        num_tracks: Number of test tracks to evaluate
        chunks_per_track: Number of chunks per track
        device: Inference device

    Returns:
        Dict with aggregated metrics:
            - StemSum/L1: L1 between summed-latent reconstruction and real sum
            - StemSum/MRSTFT: MRSTFT between the two
            - StemSum/ReconL1: L1 of oracle (encode-decode the sum directly)
            - StemSum/Rate: StemSum/L1 / StemSum/ReconL1
    """
    import os
    import json

    model.eval()

    if device is None:
        device = next(model.parameters()).device

    index_path = os.path.join(chunks_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    test_tracks = index.get("test", {})
    if not test_tracks:
        return {}

    # Select tracks
    track_keys = sorted(test_tracks.keys())[:num_tracks]

    stem_order = ["drums", "bass", "other", "vocals"]

    l1_sum_interp = 0.0
    mr_sum_interp = 0.0
    l1_sum_real = 0.0
    mr_sum_real = 0.0
    count = 0

    with torch.no_grad():
        for track_key in track_keys:
            track_info = test_tracks[track_key]
            stems = track_info["stems"]
            n_chunks = track_info["num_chunks"]

            # Check all 4 stems exist
            if not all(s in stems for s in stem_order):
                continue

            # Load all stem data
            stem_data = {}
            for stem_name in stem_order:
                path = os.path.join(chunks_dir, "test", stems[stem_name])
                data = torch.load(path, map_location="cpu", weights_only=True)
                stem_data[stem_name] = {
                    "x_stft": data["x_stft"].float(),
                    "x_wave": data["x_wave"].float(),
                }

            # Evaluate chunks
            indices = list(range(min(n_chunks, chunks_per_track)))
            for ci in indices:
                # Encode each stem
                latents = []
                waves = []
                for stem_name in stem_order:
                    x_stft = stem_data[stem_name]["x_stft"][ci].unsqueeze(0).to(device)
                    x_wave = stem_data[stem_name]["x_wave"][ci].unsqueeze(0).to(device)
                    if x_wave.dim() == 2:
                        x_wave = x_wave.unsqueeze(1)

                    with autocast("cuda", enabled=use_amp):
                        z = model.encoder(x_stft)

                    latents.append(z)
                    waves.append(x_wave)

                with autocast("cuda", enabled=use_amp):
                    # Sum latents and decode
                    z_sum = sum(latents)
                    x_from_latent_sum = model.decoder(z_sum)

                    # Real mixture (sum of waveforms)
                    x_real_mix = sum(waves)

                    # Truncate to decoder target_length
                    tgt = model.decoder.target_length
                    if x_real_mix.size(-1) > tgt:
                        x_real_mix = x_real_mix[..., :tgt]

                    # Oracle: encode the real sum, decode
                    x_real_stft = model._compute_stft_from_wave(x_real_mix)
                    z_real = model.encoder(x_real_stft)
                    x_from_real = model.decoder(z_real)

                    # Metrics
                    l1_interp = F.l1_loss(x_from_latent_sum, x_real_mix).item()
                    mr_interp = model.mrstft_loss(x_from_latent_sum, x_real_mix).item()
                    l1_real = F.l1_loss(x_from_real, x_real_mix).item()
                    mr_real = model.mrstft_loss(x_from_real, x_real_mix).item()

                l1_sum_interp += l1_interp
                mr_sum_interp += mr_interp
                l1_sum_real += l1_real
                mr_sum_real += mr_real
                count += 1

    if count == 0:
        return {}

    l1_interp_mean = l1_sum_interp / count
    mr_interp_mean = mr_sum_interp / count
    l1_real_mean = l1_sum_real / count
    mr_real_mean = mr_sum_real / count

    return {
        'StemSum/L1': l1_interp_mean,
        'StemSum/MRSTFT': mr_interp_mean,
        'StemSum/ReconL1': l1_real_mean,
        'StemSum/ReconMRSTFT': mr_real_mean,
        'StemSum/Rate': (l1_interp_mean + mr_interp_mean) / (l1_real_mean + mr_real_mean + 1e-8),
        'StemSum/NumSamples': count,
    }
