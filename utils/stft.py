"""
STFT helper functions for real/imaginary format conversions.

These utilities handle conversion between waveforms and STFT representations
using PyTorch's native STFT/ISTFT functions with real/imaginary channel format.
"""

import torch


def stft_ri(
    waveform: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = False,
) -> torch.Tensor:
    """
    Compute STFT and return real/imaginary tensor.
    
    Args:
        waveform: Input waveform [L] or [B, L]
        n_fft: FFT size
        hop_length: Hop length between frames
        win_length: Window length
        center: Whether to center-pad the signal
    
    Returns:
        STFT tensor with real/imaginary channels:
        - input [L] -> output [2, F, T]
        - input [B, L] -> output [B, 2, F, T]
    """
    if waveform.dim() == 1:
        x = waveform[None, :]  # [1, L]
        squeeze_batch = True
    elif waveform.dim() == 2:
        x = waveform
        squeeze_batch = False
    else:
        raise ValueError(f"waveform must be [L] or [B,L], got {waveform.shape}")

    x = x.float()
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )  # [B, F, T] complex

    out = torch.stack([X.real, X.imag], dim=1)  # [B, 2, F, T]
    return out[0] if squeeze_batch else out


def istft_ri(
    stft_ri_tensor: torch.Tensor,
    hop_length: int,
    win_length: int,
    length: int,
    center: bool = False,
    normalized: bool = False,
) -> torch.Tensor:
    """
    Compute inverse STFT from real/imaginary tensor.
    
    Args:
        stft_ri_tensor: STFT tensor [2, F, T] or [B, 2, F, T]
        hop_length: Hop length between frames
        win_length: Window length
        length: Desired output length
        center: Whether center padding was used in STFT
        normalized: Whether to use normalized ISTFT
    
    Returns:
        Reconstructed waveform:
        - input [2, F, T] -> output [L]
        - input [B, 2, F, T] -> output [B, L]
    """
    if stft_ri_tensor.dim() == 3:
        x = stft_ri_tensor[None, ...]  # [1, 2, F, T]
        squeeze_batch = True
    elif stft_ri_tensor.dim() == 4:
        x = stft_ri_tensor
        squeeze_batch = False
    else:
        raise ValueError(f"stft_ri must be [2,F,T] or [B,2,F,T], got {stft_ri_tensor.shape}")

    x = x.float()
    real = x[:, 0]
    imag = x[:, 1]
    X = torch.complex(real, imag)  # [B, F, T]

    n_fft = (X.shape[1] - 1) * 2
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)

    wav = torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=normalized,
        length=length,
        return_complex=False,
    )  # [B, L]
    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return wav[0] if squeeze_batch else wav

