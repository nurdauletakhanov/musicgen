"""Audio utility functions."""

from typing import Tuple

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


def compute_stft(
    waveform: torch.Tensor,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """
    Compute complex STFT and return real/imaginary components.

    Args:
        waveform: [samples] tensor (should be RMS normalized)
        window: Pre-computed Hann window tensor
        n_fft: FFT size
        hop_length: hop between frames

    Returns:
        stft_ri: [2, n_freq_bins, n_frames] tensor (real, imag)
    """
    waveform = waveform.float()
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window.size(0),
        window=window,
        return_complex=True,
        center=True,
    )
    return torch.stack([stft.real, stft.imag], dim=0)


def peak_normalize(waveform: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, float]:
    """
    Peak normalize waveform so that abs(waveform).max() == 1.

    Peak normalization is preferred over RMS normalization for this codebase:
    it guarantees every sample stays inside [-1, 1] (no clipping when the model
    tries to reproduce drum transients), it's the natural range for the iSTFT
    output, and it makes the mixing math (alpha*x1 + beta*x2) easy to bound.

    Args:
        waveform: Audio tensor [samples]
        eps: Small value to prevent division by zero on silent chunks

    Returns:
        normalized: Peak-normalized waveform (peak == 1 unless silent)
        peak: Original peak value (for potential reconstruction)
    """
    peak = waveform.abs().max().clamp_min(eps)
    return waveform / peak, peak.item()

