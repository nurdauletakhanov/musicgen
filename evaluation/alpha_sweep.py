"""
Fixed alpha sweep evaluation for linearity analysis.

This module provides controlled evaluation of latent interpolation
at fixed alpha values, separate from random alpha training metrics.

Key principle: Fixed alpha -> curves (answers "where does linearity break?")
"""

import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset


def run_alpha_sweep(
    model: torch.nn.Module,
    dataset: Dataset,
    alphas: List[float],
    num_samples: int = 100,
    device: torch.device = None,
    use_amp: bool = True,
) -> Dict[float, Dict[str, float]]:
    """
    Run fixed alpha sweep evaluation.
    
    For each alpha value, compute MixReconInterp and MixRate on fixed
    sample pairs. This is a controlled diagnostic, NOT a distribution.
    
    Args:
        model: Autoencoder model with encoder, decoder, mrstft_loss
        dataset: Validation dataset to sample from
        alphas: List of fixed alpha values (e.g., [0.1, 0.3, 0.5, 0.7, 0.9])
        num_samples: Number of sample pairs to evaluate
        device: Device for inference
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dict mapping alpha -> metrics dict with:
            - MixReconInterp: Total interpolation loss
            - MixReconInterp/WavL1: L1 component
            - MixReconInterp/MRSTFT: MRSTFT component
            - MixReconReal: Total oracle loss
            - MixRate: MixReconInterp / MixReconReal
            - MixGap: MixReconInterp - MixReconReal
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Get deterministic sample pairs
    num_samples = min(num_samples, len(dataset) // 2)
    indices = list(range(len(dataset)))
    random.seed(42)  # Deterministic for reproducibility
    random.shuffle(indices)
    
    pair_indices = [(indices[i], indices[i + num_samples]) for i in range(num_samples)]
    
    results = {}
    
    with torch.no_grad():
        for alpha in alphas:
            alpha_metrics = _evaluate_alpha(
                model=model,
                dataset=dataset,
                pair_indices=pair_indices,
                alpha=alpha,
                device=device,
                use_amp=use_amp,
            )
            results[alpha] = alpha_metrics
    
    return results


def _evaluate_alpha(
    model: torch.nn.Module,
    dataset: Dataset,
    pair_indices: List[tuple],
    alpha: float,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Evaluate a single fixed alpha value."""
    beta = 1.0 - alpha
    
    # Accumulators
    interp_l1_sum = 0.0
    interp_mr_sum = 0.0
    real_l1_sum = 0.0
    real_mr_sum = 0.0
    count = 0
    
    for idx1, idx2 in pair_indices:
        sample1 = dataset[idx1]
        sample2 = dataset[idx2]
        
        x1_stft = sample1['x_stft'].unsqueeze(0).to(device)
        x2_stft = sample2['x_stft'].unsqueeze(0).to(device)
        x1_wave = sample1['x_wave'].unsqueeze(0).to(device)
        x2_wave = sample2['x_wave'].unsqueeze(0).to(device)
        
        if x1_wave.dim() == 2:
            x1_wave = x1_wave.unsqueeze(1)
            x2_wave = x2_wave.unsqueeze(1)

        # Truncate to decoder target_length (handles padded preprocessing)
        tgt = model.decoder.target_length
        if x1_wave.size(-1) > tgt:
            x1_wave = x1_wave[..., :tgt]
            x2_wave = x2_wave[..., :tgt]

        with autocast("cuda", enabled=use_amp):
            # Encode individual samples
            z1 = model.encoder(x1_stft)
            z2 = model.encoder(x2_stft)
            
            # Mix waveforms (target)
            x_mix_wave = alpha * x1_wave + beta * x2_wave
            
            # Oracle: D(E(x_mix))
            x_mix_stft = model._compute_stft_from_wave(x_mix_wave)
            z_real = model.encoder(x_mix_stft)
            x_real_recon = model.decoder(z_real)
            
            # Interpolation: D(alpha * z1 + beta * z2)
            z_interp = alpha * z1 + beta * z2
            x_interp = model.decoder(z_interp)
            
            # Compute losses
            l1_real = F.l1_loss(x_real_recon, x_mix_wave).item()
            mr_real = model.mrstft_loss(x_real_recon, x_mix_wave).item()
            
            l1_interp = F.l1_loss(x_interp, x_mix_wave).item()
            mr_interp = model.mrstft_loss(x_interp, x_mix_wave).item()
        
        interp_l1_sum += l1_interp
        interp_mr_sum += mr_interp
        real_l1_sum += l1_real
        real_mr_sum += mr_real
        count += 1
    
    # Compute means
    interp_l1 = interp_l1_sum / count
    interp_mr = interp_mr_sum / count
    real_l1 = real_l1_sum / count
    real_mr = real_mr_sum / count
    
    # Use same weighting as training
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

