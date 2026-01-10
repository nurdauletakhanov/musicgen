"""Audio utility functions."""

import torch


def safe_peak_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Safely normalize audio to peak amplitude of 1.0.
    
    Handles NaN, inf values by replacing them with 0.0 before normalization.
    
    Args:
        x: Audio tensor of any shape
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Peak-normalized audio tensor
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    m = x.abs().max().clamp_min(eps)
    return x / m

