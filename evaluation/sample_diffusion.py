"""
Sample audio from a trained Latent Diffusion Model.

This script loads a trained diffusion model and generates audio samples.
The diffusion model produces latent vectors, which are decoded to audio
using the pre-trained autoencoder.

Usage:
    # Generate 8 samples with default settings
    python evaluation/sample_diffusion.py --config config_diffusion.yaml --checkpoint checkpoints/latent-diffusion/best_model.pth

    # Generate 16 samples with DDIM acceleration (50 steps instead of 1000)
    python evaluation/sample_diffusion.py --config config_diffusion.yaml --checkpoint checkpoints/latent-diffusion/best_model.pth --num-samples 16 --ddim --ddim-steps 50

    # Generate samples and specify output directory
    python evaluation/sample_diffusion.py --config config_diffusion.yaml --checkpoint checkpoints/latent-diffusion/best_model.pth --output ./my_samples
"""

import os
import sys
import argparse
import time

import torch
import numpy as np
import yaml
import soundfile as sf
from tqdm import tqdm

from models.diffusion import LatentDiffusion, NoiseScheduler
from models.autoencoder import Autoencoder
from training.hub_utils import resolve_checkpoint_path


def get_device(gpu_index: int = 0) -> torch.device:
    """Get CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        if gpu_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested GPU {gpu_index} but only {torch.cuda.device_count()} available."
            )
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


def load_autoencoder(config_path: str, checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load pre-trained autoencoder.
    
    Returns:
        autoencoder: The loaded model in eval mode
        sample_rate: Audio sample rate from config
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg['model']
    stft_cfg = cfg.get('stft', {})
    data_cfg = cfg['data']
    
    n_fft = stft_cfg.get('n_fft', 1024)
    sample_rate = data_cfg.get('sample_rate', 44100)
    chunk_seconds = data_cfg.get('chunk_seconds', 1.0)
    n_freq_bins = n_fft // 2 + 1
    chunk_samples = int(sample_rate * chunk_seconds)
    
    ae_config = {
        'd_model': model_cfg['d_model'],
        'n_heads': model_cfg['n_heads'],
        'n_layers': model_cfg['n_layers'],
        'num_segments': model_cfg['num_segments'],
        'n_freq_bins': n_freq_bins,
        'channels': model_cfg['channels'],
        'upsampling_factors': model_cfg['upsampling_factors'],
        'target_length': chunk_samples,
        'dropout': 0.0,
    }
    
    # Resolve checkpoint path (download from Hub if necessary)
    resolved_checkpoint = resolve_checkpoint_path(checkpoint_path)
    if resolved_checkpoint != checkpoint_path:
        print(f"Downloaded autoencoder checkpoint from Hub to: {resolved_checkpoint}")
    
    # Build model
    autoencoder = Autoencoder(**ae_config).to(device)
    
    # Load weights
    ckpt = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ckpt['model_state_dict'])
    autoencoder.eval()
    
    return autoencoder, sample_rate


def load_diffusion_model(config_path: str, checkpoint_path: str, device: torch.device, use_ema: bool = True) -> LatentDiffusion:
    """
    Load trained diffusion model.
    
    Args:
        config_path: Path to diffusion config YAML
        checkpoint_path: Path to diffusion checkpoint
        device: Device to load model on
        use_ema: Whether to use EMA weights (recommended for sampling)
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    d_cfg = cfg['diffusion']
    
    # Build model
    model = LatentDiffusion(
        d_model=d_cfg['d_model'],
        num_segments=d_cfg['num_segments'],
        n_heads=d_cfg.get('n_heads', 8),
        n_layers=d_cfg.get('n_layers', 6),
        num_timesteps=d_cfg.get('num_timesteps', 1000),
        dropout=0.0,  # No dropout during inference
        schedule=d_cfg.get('schedule', 'cosine'),
    ).to(device)
    
    # Resolve checkpoint path (download from Hub if necessary)
    resolved_checkpoint = resolve_checkpoint_path(checkpoint_path)
    if resolved_checkpoint != checkpoint_path:
        print(f"Downloaded diffusion checkpoint from Hub to: {resolved_checkpoint}")
    
    # Load weights
    ckpt = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
    
    if use_ema and 'ema_state_dict' in ckpt:
        # Load EMA weights into model
        ema_shadow = ckpt['ema_state_dict']['shadow']
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            if name in ema_shadow:
                state_dict[name] = ema_shadow[name]
        model.load_state_dict(state_dict)
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print("Loaded model weights (no EMA available)")
    
    model.eval()
    return model


@torch.no_grad()
def sample_ddpm(
    model: LatentDiffusion,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Standard DDPM sampling (all timesteps).
    
    This is the original sampling method - accurate but slow (1000 steps).
    
    Args:
        model: Trained diffusion model
        batch_size: Number of samples to generate
        device: Device to sample on
        show_progress: Show tqdm progress bar
        
    Returns:
        z: Generated latents [B, S, D]
    """
    return model.sample(batch_size, device, show_progress)


@torch.no_grad()
def sample_ddim(
    model: LatentDiffusion,
    batch_size: int,
    device: torch.device,
    num_steps: int = 50,
    eta: float = 0.0,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    DDIM sampling (accelerated, fewer steps).
    
    DDIM (Denoising Diffusion Implicit Models) allows sampling with fewer
    steps by using a deterministic update rule.
    
    Args:
        model: Trained diffusion model
        batch_size: Number of samples to generate
        device: Device to sample on
        num_steps: Number of sampling steps (e.g., 50 instead of 1000)
        eta: Stochasticity (0 = deterministic, 1 = DDPM equivalent)
        show_progress: Show tqdm progress bar
        
    Returns:
        z: Generated latents [B, S, D]
    """
    scheduler = model.scheduler
    scheduler.to(device)
    
    # Create subsequence of timesteps (evenly spaced)
    T = model.num_timesteps
    timesteps = torch.linspace(T - 1, 0, num_steps, dtype=torch.long, device=device)
    
    # Start from pure noise
    z = torch.randn(batch_size, model.num_segments, model.d_model, device=device)
    
    iterator = timesteps if not show_progress else tqdm(timesteps, desc="DDIM Sampling")
    
    for i, t in enumerate(iterator):
        t_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = model.denoiser(z, t_batch)
        
        # Get alpha values
        alpha_prod_t = scheduler.alphas_cumprod[t.long()].to(device)
        
        # Get alpha at next timestep (or 1.0 if we're at the end)
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_prod_t_prev = scheduler.alphas_cumprod[t_prev.long()].to(device)
        else:
            alpha_prod_t_prev = torch.tensor(1.0, device=device)
        
        # Predict x_0 from noise
        sqrt_alpha = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_prod_t)
        z_0_pred = (z - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        
        # Compute "direction pointing to x_t"
        sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prod_t_prev)
        
        # Compute sigma for stochasticity
        sigma = eta * torch.sqrt(
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        )
        
        # DDIM update
        z = (
            torch.sqrt(alpha_prod_t_prev) * z_0_pred +
            torch.sqrt(1 - alpha_prod_t_prev - sigma**2) * noise_pred +
            sigma * torch.randn_like(z)
        )
    
    return z


def save_samples(
    audio_samples: torch.Tensor,
    output_dir: str,
    sample_rate: int,
    prefix: str = "sample",
):
    """Save audio samples to WAV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(audio_samples.size(0)):
        audio = audio_samples[i, 0].cpu().numpy()
        
        # Normalize to prevent clipping
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.wav")
        sf.write(path, audio, sample_rate, subtype="FLOAT")
    
    print(f"Saved {audio_samples.size(0)} samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sample from trained Latent Diffusion Model")
    parser.add_argument("--config", type=str, required=True, help="Path to diffusion config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to diffusion checkpoint")
    parser.add_argument("--output", type=str, default="./generated_samples", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (defaults to num-samples)")
    parser.add_argument("--ddim", action="store_true", help="Use DDIM sampling (faster)")
    parser.add_argument("--ddim-steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0=deterministic)")
    parser.add_argument("--no-ema", action="store_true", help="Don't use EMA weights")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Device
    device = get_device(args.gpu)
    print(f"Using device: {device}")
    
    # Load configs
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    ae_config_path = cfg['autoencoder']['config']
    ae_checkpoint_path = cfg['autoencoder']['checkpoint']
    
    # Load models
    print("Loading autoencoder...")
    autoencoder, sample_rate = load_autoencoder(ae_config_path, ae_checkpoint_path, device)
    
    print("Loading diffusion model...")
    diffusion = load_diffusion_model(args.config, args.checkpoint, device, use_ema=not args.no_ema)
    
    # Sampling
    num_samples = args.num_samples
    batch_size = args.batch_size or num_samples
    
    all_audio = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        print(f"\nGenerating batch {batch_idx + 1}/{num_batches} ({current_batch_size} samples)...")
        
        # Sample latents
        if args.ddim:
            z = sample_ddim(
                diffusion,
                current_batch_size,
                device,
                num_steps=args.ddim_steps,
                eta=args.eta,
            )
        else:
            z = sample_ddpm(diffusion, current_batch_size, device)
        
        # Decode to audio
        print("Decoding to audio...")
        audio = autoencoder.decoder(z)  # [B, 1, L]
        all_audio.append(audio)
    
    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed:.1f}s ({elapsed/num_samples:.2f}s per sample)")
    
    # Concatenate and save
    all_audio = torch.cat(all_audio, dim=0)
    save_samples(all_audio, args.output, sample_rate)
    
    print(f"\nDone! Generated {num_samples} samples at {sample_rate}Hz")


if __name__ == "__main__":
    main()

