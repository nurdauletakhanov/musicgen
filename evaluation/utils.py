"""Shared evaluation utilities: model loading, test data loading, metrics, and log parsing."""

import json
import os
import random
import re
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from models.autoencoder import Autoencoder


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Args:
        estimate: Reconstructed waveform, shape (B, 1, T) or (B, T).
        reference: Ground-truth waveform, same shape.
        eps: Small constant for numerical stability.

    Returns:
        Mean SI-SDR in dB across the batch.
    """
    estimate = estimate.reshape(estimate.shape[0], -1)
    reference = reference.reshape(reference.shape[0], -1)

    dot = torch.sum(estimate * reference, dim=-1, keepdim=True)
    s_ref_sq = torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
    proj = (dot / s_ref_sq) * reference

    noise = estimate - proj
    si_sdr_val = 10 * torch.log10(
        torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps) + eps
    )
    return float(si_sdr_val.mean().item())


def load_model(checkpoint_path: str, device: torch.device) -> Autoencoder:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(ckpt["model_config"])
    # Remove deprecated keys from old checkpoints
    for key in ("stft_loss_weight", "l1_weight"):
        config.pop(key, None)
    model = Autoencoder(**config)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if result.missing_keys:
        print(f"Warning: missing keys in checkpoint: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Warning: unexpected keys in checkpoint: {result.unexpected_keys}")
    model.to(device).eval()
    return model


STEM_ORDER = ["drums", "bass", "other", "vocals"]

STEM_PAIRS = [
    ("drums", "vocals"),
    ("drums", "bass"),
    ("bass", "vocals"),
    ("other", "vocals"),
    ("drums", "other"),
    ("bass", "other"),
]

MAX_CHUNKS_PER_TRACK = 5


def _load_test_tracks(chunks_dir: str) -> Dict:
    """Load and validate test tracks from index.json."""
    index_path = os.path.join(chunks_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    test_tracks = index.get("test", {})
    if not test_tracks:
        raise RuntimeError("No test data found in index.json")
    return test_tracks


def load_test_stem_pairs(chunks_dir: str, num_samples: int = 100) -> List[Dict]:
    """
    Load stem pairs from MUSDB test split.

    Returns list of dicts with x1_stft, x1_wave, x2_stft, x2_wave.
    """
    test_tracks = _load_test_tracks(chunks_dir)
    samples = []

    for track_key, track_info in test_tracks.items():
        if not isinstance(track_info, dict) or "stems" not in track_info:
            continue

        stems = track_info["stems"]
        n_chunks = track_info.get("num_chunks", 0)

        if not all(s in stems for s in STEM_ORDER):
            continue

        # Load all stem data for this track
        stem_data = {}
        for stem_name in STEM_ORDER:
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

        for chunk_idx in range(min(n_chunks, MAX_CHUNKS_PER_TRACK)):
            for stem1_name, stem2_name in STEM_PAIRS:
                if stem1_name not in stem_data or stem2_name not in stem_data:
                    continue
                samples.append({
                    "x1_stft": stem_data[stem1_name]["x_stft"][chunk_idx],
                    "x1_wave": stem_data[stem1_name]["x_wave"][chunk_idx],
                    "x2_stft": stem_data[stem2_name]["x_stft"][chunk_idx],
                    "x2_wave": stem_data[stem2_name]["x_wave"][chunk_idx],
                    "track": track_key,
                    "stems": f"{stem1_name}+{stem2_name}",
                })

    random.seed(42)
    random.shuffle(samples)
    return samples[:num_samples]


def load_test_singles(chunks_dir: str, num_samples: int = 200) -> List[Dict]:
    """Load individual stem chunks from MUSDB test split for reconstruction eval."""
    test_tracks = _load_test_tracks(chunks_dir)
    samples = []

    for track_key, track_info in test_tracks.items():
        if not isinstance(track_info, dict) or "stems" not in track_info:
            continue

        stems = track_info["stems"]
        n_chunks = track_info.get("num_chunks", 0)

        for stem_name in STEM_ORDER:
            if stem_name not in stems:
                continue
            path = os.path.join(chunks_dir, "test", stems[stem_name])
            if not os.path.exists(path):
                continue
            data = torch.load(path, map_location="cpu", weights_only=True)

            for chunk_idx in range(min(n_chunks, MAX_CHUNKS_PER_TRACK)):
                samples.append({
                    "x_stft": data["x_stft"][chunk_idx].float(),
                    "x_wave": data["x_wave"][chunk_idx].float(),
                    "track": track_key,
                    "stem": stem_name,
                })

    random.seed(42)
    random.shuffle(samples)
    return samples[:num_samples]



def evaluate_alpha_on_pairs(
    model: torch.nn.Module,
    pairs: List[Dict],
    alpha: float,
    device: torch.device,
    per_sample_stats: bool = False,
) -> Dict[str, float]:
    """Core alpha evaluation: encode, interpolate, compare to oracle.

    Each pair dict must have keys: x1_stft, x1_wave, x2_stft, x2_wave.

    Args:
        model: Autoencoder with encoder, decoder, mrstft_loss.
        pairs: List of sample pair dicts.
        alpha: Mixing coefficient (beta = 1 - alpha).
        device: Inference device.
        per_sample_stats: If True, return std/median/p90 in addition to means.

    Returns:
        Dict with MixReconInterp, MixReconReal, MixRate, MixGap, and
        optionally per-sample statistics (MixRate_std, MixRate_median, MixRate_p90).
    """
    beta = 1.0 - alpha

    interp_l1_list = []
    interp_mr_list = []
    real_l1_list = []
    real_mr_list = []

    for sample in pairs:
        x1_stft = sample["x1_stft"].unsqueeze(0).to(device)
        x1_wave = sample["x1_wave"].unsqueeze(0).to(device)
        x2_stft = sample["x2_stft"].unsqueeze(0).to(device)
        x2_wave = sample["x2_wave"].unsqueeze(0).to(device)

        if x1_wave.dim() == 2:
            x1_wave = x1_wave.unsqueeze(1)
            x2_wave = x2_wave.unsqueeze(1)

        tgt = model.decoder.target_length
        if x1_wave.size(-1) > tgt:
            x1_wave = x1_wave[..., :tgt]
            x2_wave = x2_wave[..., :tgt]

        z1 = model.encoder(x1_stft)
        z2 = model.encoder(x2_stft)

        x_mix_wave = alpha * x1_wave + beta * x2_wave

        x_mix_stft = model._compute_stft_from_wave(x_mix_wave)
        z_real = model.encoder(x_mix_stft)
        x_real_recon, _ = model.decoder(z_real)

        z_interp = alpha * z1 + beta * z2
        x_interp, _ = model.decoder(z_interp)

        l1_real = F.l1_loss(x_real_recon, x_mix_wave).item()
        mr_real = model.mrstft_loss(x_real_recon, x_mix_wave).item()
        l1_interp = F.l1_loss(x_interp, x_mix_wave).item()
        mr_interp = model.mrstft_loss(x_interp, x_mix_wave).item()

        interp_l1_list.append(l1_interp)
        interp_mr_list.append(mr_interp)
        real_l1_list.append(l1_real)
        real_mr_list.append(mr_real)

    if not hasattr(model, 'mix_l1_weight') or not hasattr(model, 'mix_mrstft_weight'):
        print("Warning: model missing mix_l1_weight/mix_mrstft_weight, using defaults (1.0)")
    mix_l1_weight = getattr(model, 'mix_l1_weight', 1.0)
    mix_mrstft_weight = getattr(model, 'mix_mrstft_weight', 1.0)

    interp_totals = [mix_l1_weight * l1 + mix_mrstft_weight * mr
                     for l1, mr in zip(interp_l1_list, interp_mr_list)]
    real_totals = [mix_l1_weight * l1 + mix_mrstft_weight * mr
                   for l1, mr in zip(real_l1_list, real_mr_list)]
    rates = [i / (r + 1e-8) for i, r in zip(interp_totals, real_totals)]

    result = {
        'MixReconInterp': float(np.mean(interp_totals)),
        'MixReconInterp/WavL1': float(np.mean(interp_l1_list)),
        'MixReconInterp/MRSTFT': float(np.mean(interp_mr_list)),
        'MixReconReal': float(np.mean(real_totals)),
        'MixReconReal/WavL1': float(np.mean(real_l1_list)),
        'MixReconReal/MRSTFT': float(np.mean(real_mr_list)),
        'MixRate': float(np.mean(rates)),
        'MixGap': float(np.mean(interp_totals)) - float(np.mean(real_totals)),
    }

    if per_sample_stats:
        result.update({
            'MixReconInterp_mean': result['MixReconInterp'],
            'MixReconInterp_std': float(np.std(interp_totals)),
            'MixReconReal_mean': result['MixReconReal'],
            'MixReconReal_std': float(np.std(real_totals)),
            'MixRate_mean': result['MixRate'],
            'MixRate_std': float(np.std(rates)),
            'MixRate_median': float(np.median(rates)),
            'MixRate_p90': float(np.percentile(rates, 90)),
            'num_samples': len(pairs),
        })

    return result


# ---------------------------------------------------------------------------
# Log parsing utilities
# ---------------------------------------------------------------------------

def parse_log_file(log_path: str, selected_epoch: int = None) -> Dict:
    """Parse training log file to extract per-epoch metrics and alpha sweeps.

    Args:
        log_path: Path to a .log file.
        selected_epoch: If provided, also return the data for this epoch.

    Returns:
        Dict with keys: epochs, alpha_sweeps, best_epoch, selected_epoch, final_epoch.
    """
    epochs = []
    alpha_sweeps = {}

    with open(log_path, "r") as f:
        lines = f.readlines()

    current_epoch = None
    epoch_data = {}
    in_alpha_sweep = False
    alpha_sweep_epoch = None

    for line in lines:
        epoch_match = re.match(r"=== Epoch (\d+)/(\d+) ===", line)
        if epoch_match:
            if current_epoch is not None and epoch_data:
                epochs.append(epoch_data)
            current_epoch = int(epoch_match.group(1))
            epoch_data = {"epoch": current_epoch}
            in_alpha_sweep = False
            continue

        # Validation metrics with mixing
        val_match = re.match(
            r"Val   - Loss: ([\d.]+) \| Recon: ([\d.]+) \(L1: ([\d.]+), MR: ([\d.]+)\) "
            r"\| MixInterp: ([\d.]+), MixReal: ([\d.]+) \| Rate: ([\d.]+), Gap: ([-\d.]+)",
            line
        )
        if val_match:
            epoch_data["val_loss"] = float(val_match.group(1))
            epoch_data["val_recon"] = float(val_match.group(2))
            epoch_data["val_l1"] = float(val_match.group(3))
            epoch_data["val_mrstft"] = float(val_match.group(4))
            epoch_data["val_mix_rate"] = float(val_match.group(7))
            continue

        # Phase 1 style (no mixing metrics)
        val_simple_match = re.match(
            r"Val   - Loss: ([\d.]+) \| Recon: ([\d.]+) \(L1: ([\d.]+), MR: ([\d.]+)\)",
            line
        )
        if val_simple_match and "val_loss" not in epoch_data:
            epoch_data["val_loss"] = float(val_simple_match.group(1))
            epoch_data["val_recon"] = float(val_simple_match.group(2))
            epoch_data["val_l1"] = float(val_simple_match.group(3))
            epoch_data["val_mrstft"] = float(val_simple_match.group(4))
            continue

        # Alpha sweep
        if "Running alpha sweep evaluation" in line:
            in_alpha_sweep = True
            alpha_sweep_epoch = current_epoch
            if alpha_sweep_epoch not in alpha_sweeps:
                alpha_sweeps[alpha_sweep_epoch] = {}
            continue

        if in_alpha_sweep:
            alpha_match = re.match(
                r"\s+alpha=([\d.]+): MixReconInterp=([\d.]+), MixRate=([\d.]+)", line
            )
            if alpha_match:
                alpha = float(alpha_match.group(1))
                alpha_sweeps[alpha_sweep_epoch][alpha] = {
                    "mix_recon_interp": float(alpha_match.group(2)),
                    "mix_rate": float(alpha_match.group(3)),
                }

    if current_epoch is not None and epoch_data:
        epochs.append(epoch_data)

    best_epoch = None
    best_loss = float("inf")
    sel_epoch = None
    for e in epochs:
        if e.get("val_loss", float("inf")) < best_loss:
            best_loss = e["val_loss"]
            best_epoch = e
        if selected_epoch is not None and e.get("epoch") == selected_epoch:
            sel_epoch = e

    return {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "selected_epoch": sel_epoch,
        "final_epoch": epochs[-1] if epochs else None,
        "alpha_sweeps": alpha_sweeps,
    }


def get_checkpoint_metrics(checkpoint_dir: str, selected_epoch: int = None) -> Dict:
    """Extract metrics from all log files in a checkpoint directory.

    Args:
        checkpoint_dir: Directory containing .log files and checkpoints.
        selected_epoch: If provided, also return the data for this epoch.

    Returns:
        Dict with keys: epochs, alpha_sweeps, best_epoch, selected_epoch, final_epoch.
    """
    log_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".log")]
    if not log_files:
        return {"epochs": [], "alpha_sweeps": {}}

    all_epochs = []
    all_sweeps = {}
    for log_file in sorted(log_files):
        log_path = os.path.join(checkpoint_dir, log_file)
        parsed = parse_log_file(log_path, selected_epoch)
        all_epochs.extend(parsed["epochs"])
        all_sweeps.update(parsed["alpha_sweeps"])

    best_epoch = None
    best_loss = float("inf")
    sel_epoch = None
    for e in all_epochs:
        if e.get("val_loss", float("inf")) < best_loss:
            best_loss = e["val_loss"]
            best_epoch = e
        if selected_epoch is not None and e.get("epoch") == selected_epoch:
            sel_epoch = e

    return {
        "epochs": all_epochs,
        "best_epoch": best_epoch,
        "selected_epoch": sel_epoch,
        "final_epoch": all_epochs[-1] if all_epochs else None,
        "alpha_sweeps": all_sweeps,
    }
