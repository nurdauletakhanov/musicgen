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


def rms_normalize(waveform: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, float]:
    """
    RMS normalize waveform for consistent amplitude across tracks.

    Args:
        waveform: Audio tensor [samples]
        eps: Small value to prevent division by zero

    Returns:
        normalized: Normalized waveform
        rms: Original RMS value (for potential reconstruction)
    """
    rms = waveform.pow(2).mean().sqrt().clamp_min(eps)
    return waveform / rms, rms.item()

