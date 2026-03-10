"""Audio sample generation for model evaluation during training."""

import os
import random
from typing import TYPE_CHECKING, Optional

import soundfile as sf
import torch
from torch.amp import autocast
from torch.nn import functional as F
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from training.tb_logger import TBLogger


def save_test_samples(
    model: torch.nn.Module,
    dataset: Dataset,
    save_dir: str,
    epoch: int,
    sample_rate: int,
    chunk_samples: int,
    num_samples: int = 5,
    device: torch.device = None,
    use_amp: bool = True,
    logger: Optional[object] = None,
    tb_logger: Optional["TBLogger"] = None,
):
    """
    Save original and reconstructed audio samples for evaluation.
    
    Args:
        model: Autoencoder model with encoder and decoder
        dataset: Dataset to sample from (validation set)
        save_dir: Base directory for saving samples
        epoch: Current epoch number (0-indexed)
        sample_rate: Audio sample rate for saving
        chunk_samples: Expected number of samples per chunk
        num_samples: Number of samples to save
        device: Device to run inference on
        use_amp: Whether to use automatic mixed precision
        logger: Optional logger for info messages
        tb_logger: Optional TensorBoard logger for audio samples
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    samples_dir = os.path.join(save_dir, "samples", f"epoch_{epoch+1}")
    os.makedirs(samples_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    if logger:
        logger.info(f"Saving {num_samples} test samples to {samples_dir}")
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            x_stft = sample["x_stft"].unsqueeze(0).to(device)
            x_wave = sample["x_wave"].unsqueeze(0).to(device)
            
            if x_wave.dim() == 2:
                x_wave = x_wave.unsqueeze(1)
            
            with autocast("cuda", enabled=use_amp):
                z = model.encoder(x_stft)
                y_hat = model.decoder(z)
            
            x_wave_orig = x_wave[0, 0].cpu().float()
            y_hat_recon = y_hat[0, 0].cpu().float()

            # Ensure lengths match chunk_samples
            x_wave_orig = _ensure_length(x_wave_orig, chunk_samples)
            y_hat_recon = _ensure_length(y_hat_recon, chunk_samples)

            # Peak-normalize to avoid clipping (RMS-normalized audio has peaks >> 1.0)
            x_wave_orig = _peak_normalize(x_wave_orig)
            y_hat_recon = _peak_normalize(y_hat_recon)

            orig_path = os.path.join(samples_dir, f"sample_{i:03d}_original.wav")
            recon_path = os.path.join(samples_dir, f"sample_{i:03d}_reconstructed.wav")

            sf.write(orig_path, x_wave_orig.numpy(), sample_rate, subtype="FLOAT")
            sf.write(recon_path, y_hat_recon.numpy(), sample_rate, subtype="FLOAT")

            # Log to TensorBoard if enabled
            if tb_logger is not None:
                tb_logger.log_audio(
                    f"sample_{i:03d}/original",
                    x_wave_orig,
                    epoch,
                    sample_rate,
                    prefix="audio"
                )
                tb_logger.log_audio(
                    f"sample_{i:03d}/reconstructed",
                    y_hat_recon,
                    epoch,
                    sample_rate,
                    prefix="audio"
                )
    
    if logger:
        logger.info(f"Test samples saved to {samples_dir}")
    
    if tb_logger is not None:
        tb_logger.flush()


def _peak_normalize(tensor: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """Normalize peak amplitude to target_peak to prevent clipping on playback."""
    peak = tensor.abs().max()
    if peak > target_peak:
        tensor = tensor / peak * target_peak
    return tensor


def _ensure_length(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    """Trim or pad tensor to target length."""
    if tensor.numel() > target_length:
        return tensor[:target_length]
    elif tensor.numel() < target_length:
        return F.pad(tensor, (0, target_length - tensor.numel()))
    return tensor

