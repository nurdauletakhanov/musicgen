"""
Fixed alpha sweep evaluation for linearity analysis.

This module provides controlled evaluation of latent interpolation
at fixed alpha values, separate from random alpha training metrics.

Key principle: Fixed alpha -> curves (answers "where does linearity break?")
"""

import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from evaluation.utils import evaluate_alpha_on_pairs


def run_alpha_sweep(
    model: torch.nn.Module,
    dataset: Dataset,
    alphas: List[float],
    num_samples: int = 100,
    device: torch.device = None,
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

    # Build pairs list in the format expected by evaluate_alpha_on_pairs
    pairs = []
    for idx1, idx2 in pair_indices:
        s1 = dataset[idx1]
        s2 = dataset[idx2]
        pairs.append({
            "x1_stft": s1['x_stft'],
            "x1_wave": s1['x_wave'],
            "x2_stft": s2['x_stft'],
            "x2_wave": s2['x_wave'],
        })

    results = {}
    with torch.no_grad():
        for alpha in alphas:
            results[alpha] = evaluate_alpha_on_pairs(
                model=model, pairs=pairs, alpha=alpha, device=device,
            )

    return results
