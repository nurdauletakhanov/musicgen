"""
Training script for Latent Diffusion Model.

This script trains a diffusion model to generate latents in the space learned
by a pre-trained autoencoder. The training process:

1. Load pre-trained autoencoder (frozen)
2. Encode training audio to latent space
3. Train diffusion model to denoise random noise into valid latents
4. Periodically generate samples by decoding diffusion outputs

Usage:
    python training/train_diffusion.py --config config_diffusion.yaml
    python training/train_diffusion.py --config config_diffusion.yaml --resume checkpoints/latent-diffusion/checkpoint_10.pth
"""

import os
import sys
import argparse
import copy

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import soundfile as sf
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from models.diffusion import LatentDiffusion, NoiseScheduler, Denoiser
from models.autoencoder import Autoencoder
from training.utils import print_logger
from data.dataloader import build_dataloaders, STFTChunkDataset, ShardedSampler


def get_device(gpu_index: int = 0) -> torch.device:
    """Get CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        if gpu_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested GPU {gpu_index} but only {torch.cuda.device_count()} available."
            )
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


class EMA:
    """
    Exponential Moving Average for model weights.
    
    EMA maintains a smoothed version of model weights, which often produces
    better samples than the raw trained weights. The update rule:
    
        ema_weights = decay * ema_weights + (1 - decay) * model_weights
    
    High decay (0.9999) means very slow updates, keeping a long history.
    """
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights as copy of model weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: torch.nn.Module):
        """Update shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: torch.nn.Module):
        """Replace model weights with EMA shadow weights (for inference)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: torch.nn.Module):
        """Restore original model weights after inference."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class LatentDataset(torch.utils.data.Dataset):
    """
    Dataset that provides pre-computed latent vectors.
    
    For faster training, we encode all audio to latents once and cache them.
    This avoids running the encoder every training step.
    
    Each .pt file contains batched latents [N, S, D] for N chunks.
    """
    
    def __init__(self, latents_dir: str, split: str = 'train'):
        from bisect import bisect_right
        import json
        
        self.latents_dir = os.path.join(latents_dir, split)
        
        # Load index with chunk counts
        latent_index_path = os.path.join(latents_dir, 'index.json')
        
        if os.path.exists(latent_index_path):
            with open(latent_index_path, 'r') as f:
                index = json.load(f)
            split_index = index.get(split, {})
        else:
            raise FileNotFoundError(f"No index found at {latent_index_path}. Run --cache-latents first.")
        
        # Build file list with cumulative chunk counts
        self.files = []
        total = 0
        for filename, count in sorted(split_index.items()):
            path = os.path.join(self.latents_dir, filename)
            if not os.path.exists(path):
                continue
            count = int(count)
            start = total
            total += count
            self.files.append({"path": path, "count": count, "start": start, "end": total})
        
        self.total_chunks = total
        self._ends = [f["end"] for f in self.files]
        self.file_count = len(self.files)  # For ShardedSampler compatibility
        
        # Cache for sequential access
        self._cache_path = None
        self._cache_data = None
    
    def __len__(self):
        return self.total_chunks
    
    def __getitem__(self, idx):
        from bisect import bisect_right
        
        if idx < 0 or idx >= self.total_chunks:
            raise IndexError(f"Index {idx} out of range for dataset length {self.total_chunks}")
        
        # Find which file contains this index
        file_idx = bisect_right(self._ends, idx)
        start = 0 if file_idx == 0 else self._ends[file_idx - 1]
        local_idx = idx - start
        
        file_entry = self.files[file_idx]
        
        # Load file (with caching)
        if file_entry["path"] != self._cache_path:
            self._cache_data = torch.load(file_entry["path"], map_location="cpu", weights_only=True)
            self._cache_path = file_entry["path"]
        
        return self._cache_data['z'][local_idx].contiguous()  # [S, D]


class DiffusionTrainer:
    """
    Trainer for latent diffusion model.
    """
    
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.cfg = cfg
        train_cfg = cfg['train']
        diffusion_cfg = cfg['diffusion']
        data_cfg = cfg['data']
        autoencoder_cfg = cfg['autoencoder']
        
        # Device setup
        self.device = get_device(train_cfg.get('gpu_index', 0))
        
        # Training params
        self.batch_size = train_cfg['batch_size']
        self.num_epochs = train_cfg['num_epochs']
        self.learning_rate = train_cfg['learning_rate']
        self.weight_decay = train_cfg.get('weight_decay', 0.0001)
        self.warmup_steps = train_cfg.get('warmup_steps', 1000)
        self.save_interval = train_cfg.get('save_interval', 5)
        self.num_workers = train_cfg.get('num_workers', 4)
        self.pin_memory = train_cfg.get('pin_memory', True)
        self.persistent_workers = train_cfg.get('persistent_workers', True)
        self.prefetch_factor = train_cfg.get('prefetch_factor', 2)
        
        # EMA settings
        self.use_ema = train_cfg.get('use_ema', True)
        self.ema_decay = train_cfg.get('ema_decay', 0.9999)
        
        # Save path
        name = cfg.get('name', 'latent-diffusion')
        base_path = train_cfg.get('save_path', './checkpoints/latent-diffusion')
        self.save_path = base_path
        os.makedirs(self.save_path, exist_ok=True)
        
        # Sampling settings
        sampling_cfg = cfg.get('sampling', {})
        self.num_samples = sampling_cfg.get('num_samples', 4)
        self.sample_interval = sampling_cfg.get('sample_interval', 10)
        self.samples_dir = sampling_cfg.get('output_dir', os.path.join(self.save_path, 'samples'))
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # Logger
        self.logs = print_logger(self.save_path)
        self.logs.info(f"Device: {self.device}")
        if self.device.type == "cuda":
            self.logs.info(f"GPU: {torch.cuda.get_device_name(self.device)}")
        
        # Store configs for model building
        self.diffusion_cfg = diffusion_cfg
        self.data_cfg = data_cfg
        self.autoencoder_cfg = autoencoder_cfg
        
        # AMP setup
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.logs.info(f"Mixed precision (AMP): {'enabled' if self.use_amp else 'disabled'}")
        
        # Track global step for warmup
        self.global_step = 0
        
        # Latent normalization statistics (computed from training data)
        # These transform latents to N(0,1) before diffusion
        # Shape: [1, 1, D] - global per-dimension to preserve temporal correlations
        self.z_mean = None
        self.z_std = None
    
    def load_autoencoder(self):
        """Load pre-trained autoencoder (frozen)."""
        # Load autoencoder config
        ae_config_path = self.autoencoder_cfg['config']
        with open(ae_config_path, 'r') as f:
            ae_cfg = yaml.safe_load(f)
        
        # Extract model config
        model_cfg = ae_cfg['model']
        stft_cfg = ae_cfg.get('stft', {})
        data_cfg = ae_cfg['data']
        
        n_fft = stft_cfg.get('n_fft', 1024)
        sample_rate = data_cfg.get('sample_rate', 44100)
        chunk_seconds = data_cfg.get('chunk_seconds', 1.0)
        n_freq_bins = n_fft // 2 + 1
        chunk_samples = int(sample_rate * chunk_seconds)
        
        ae_model_config = {
            'd_model': model_cfg['d_model'],
            'n_heads': model_cfg['n_heads'],
            'n_layers': model_cfg['n_layers'],
            'num_segments': model_cfg['num_segments'],
            'n_freq_bins': n_freq_bins,
            'channels': model_cfg['channels'],
            'upsampling_factors': model_cfg['upsampling_factors'],
            'target_length': chunk_samples,
            'dropout': 0.0,  # No dropout during inference
        }
        
        # Build and load autoencoder
        self.autoencoder = Autoencoder(**ae_model_config).to(self.device)
        
        ckpt_path = self.autoencoder_cfg['checkpoint']
        self.logs.info(f"Loading autoencoder from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.autoencoder.load_state_dict(ckpt['model_state_dict'])
        
        # Freeze autoencoder
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        self.logs.info("Autoencoder loaded and frozen")
        
        # Store for sampling
        self.sample_rate = sample_rate
        
        return ae_model_config
    
    def build_dataloaders(self, use_cached_latents: bool = False):
        """Build data loaders using ShardedSampler for memory-efficient access."""
        chunks_dir = self.data_cfg['chunks_dir']
        train_split = self.data_cfg.get('split', 'train')
        val_split = self.data_cfg.get('val_split', 'validation')
        
        self.use_cached_latents = use_cached_latents
        
        if use_cached_latents:
            # Use pre-computed latents with ShardedSampler (same pattern as STFT loader)
            latents_dir = self.data_cfg['latents_cache_dir']
            self.train_dataset = LatentDataset(latents_dir, train_split)
            self.val_dataset = LatentDataset(latents_dir, val_split)
            
            # Use ShardedSampler for sequential file access (same as STFT loader)
            self.train_sampler = ShardedSampler(
                self.train_dataset,
                shuffle_files=True,
                shuffle_within_file=True,
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                drop_last=True,
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
            )
        else:
            # Use existing dataloader infrastructure with ShardedSampler
            # This ensures sequential file access and minimal RAM usage
            (
                self.train_loader,
                self.val_loader,
                self.train_dataset,
                self.val_dataset,
                self.train_sampler,
            ) = build_dataloaders(
                chunks_dir=chunks_dir,
                train_split=train_split,
                val_split=val_split,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                device=self.device,
                latent_mix_weight=0.0,  # No mixing for diffusion
                decode_mix_weight=0.0,
                logger=self.logs,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
            )
        
        self.logs.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
    
    def compute_latent_statistics(self):
        """
        Compute mean and std of latent space from training data.
        
        This is essential for latent diffusion: we normalize latents to N(0,1)
        before training diffusion, then denormalize after sampling.
        
        Statistics are computed GLOBALLY per-dimension [1, 1, D]:
          μ_d = E_{x,s}[z_{s,d}],  σ_d = sqrt(Var_{x,s}[z_{s,d}])
        
        This preserves temporal correlation structure because we don't re-center
        each time position independently.
        """
        self.logs.info("Computing latent statistics from training data...")
        self.logs.info("Using GLOBAL per-dimension normalization [1, 1, D] to preserve temporal structure")
        
        # Use parallel/batch Welford's algorithm for speed
        # Statistics computed over all B×S tokens, keeping only D dimension
        count = 0
        mean = None
        M2 = None  # Sum of squared differences from mean
        
        # Iterate over training data
        for batch in tqdm(self.train_loader, desc="Computing stats", leave=False):
            if self.use_cached_latents:
                z = batch.to(self.device, non_blocking=True)  # [B, S, D]
            else:
                z = self.encode_batch(batch)
            
            B, S, D = z.shape
            
            # Flatten B and S dimensions: treat each token as a sample
            z_flat = z.reshape(-1, D)  # [B*S, D]
            n = z_flat.size(0)
            
            if mean is None:
                # Initialize with batch statistics
                mean = z_flat.mean(dim=0)  # [D]
                M2 = ((z_flat - mean) ** 2).sum(dim=0)  # [D]
                count = n
            else:
                # Parallel Welford update for the entire batch
                batch_mean = z_flat.mean(dim=0)  # [D]
                batch_var = z_flat.var(dim=0, unbiased=False)  # [D]
                
                # Combined mean
                delta = batch_mean - mean
                new_count = count + n
                new_mean = mean + delta * n / new_count
                
                # Combined M2 (Chan's parallel algorithm)
                M2 = M2 + batch_var * n + delta ** 2 * count * n / new_count
                
                mean = new_mean
                count = new_count
        
        # Compute final std
        variance = M2 / count
        std = torch.sqrt(variance + 1e-8)  # Add epsilon for numerical stability
        
        # Reshape to [1, 1, D] for broadcasting with [B, S, D] tensors
        self.z_mean = mean.view(1, 1, -1).to(self.device)  # [1, 1, D]
        self.z_std = std.view(1, 1, -1).to(self.device)    # [1, 1, D]
        
        # Log statistics
        total_tokens = count
        self.logs.info(f"Latent statistics computed from {total_tokens:,} tokens:")
        self.logs.info(f"  Shape: {self.z_mean.shape} (global per-dimension)")
        self.logs.info(f"  Mean: min={self.z_mean.min().item():.4f}, max={self.z_mean.max().item():.4f}, avg={self.z_mean.mean().item():.4f}")
        self.logs.info(f"  Std:  min={self.z_std.min().item():.4f}, max={self.z_std.max().item():.4f}, avg={self.z_std.mean().item():.4f}")
        
        # Save statistics to file for reuse
        stats_path = os.path.join(self.save_path, 'latent_stats.pt')
        torch.save({
            'z_mean': self.z_mean.cpu(),
            'z_std': self.z_std.cpu(),
            'count': count,
        }, stats_path)
        self.logs.info(f"Latent statistics saved to {stats_path}")
    
    def load_latent_statistics(self, stats_path: str = None):
        """Load pre-computed latent statistics."""
        if stats_path is None:
            stats_path = os.path.join(self.save_path, 'latent_stats.pt')
        
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location=self.device, weights_only=True)
            self.z_mean = stats['z_mean'].to(self.device)
            self.z_std = stats['z_std'].to(self.device)
            self.logs.info(f"Loaded latent statistics from {stats_path}")
            self.logs.info(f"  Shape: {self.z_mean.shape} ({'global per-dim' if self.z_mean.shape[1] == 1 else 'per-position (OLD)'})")
            self.logs.info(f"  Mean: avg={self.z_mean.mean().item():.4f}")
            self.logs.info(f"  Std:  avg={self.z_std.mean().item():.4f}")
            return True
        return False
    
    def normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize latents to N(0,1) using pre-computed statistics."""
        if self.z_mean is None or self.z_std is None:
            raise RuntimeError("Latent statistics not computed. Call compute_latent_statistics() first.")
        return (z - self.z_mean) / self.z_std
    
    def denormalize_latents(self, z_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize latents back to original scale."""
        if self.z_mean is None or self.z_std is None:
            raise RuntimeError("Latent statistics not computed. Call compute_latent_statistics() first.")
        return z_normalized * self.z_std + self.z_mean
    
    def build_model(self):
        """Build diffusion model, optimizer, and EMA."""
        d_cfg = self.diffusion_cfg
        
        # Build diffusion model
        self.model = LatentDiffusion(
            d_model=d_cfg['d_model'],
            num_segments=d_cfg['num_segments'],
            n_heads=d_cfg.get('n_heads', 8),
            n_layers=d_cfg.get('n_layers', 6),
            num_timesteps=d_cfg.get('num_timesteps', 1000),
            dropout=d_cfg.get('dropout', 0.1),
            schedule=d_cfg.get('schedule', 'cosine'),
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # EMA
        if self.use_ema:
            self.ema = EMA(self.model, decay=self.ema_decay)
        
        # Log parameters
        num_params = self.model.denoiser.num_parameters()
        self.logs.info(f"Diffusion model parameters: {num_params:,}")
    
    def get_lr_scale(self) -> float:
        """Linear warmup learning rate scale."""
        if self.global_step < self.warmup_steps:
            return self.global_step / max(1, self.warmup_steps)
        return 1.0
    
    def encode_batch(self, batch: dict) -> torch.Tensor:
        """Encode a batch of STFT data to latents using frozen autoencoder."""
        x_stft = batch['x_stft'].to(self.device, non_blocking=True)
        
        with torch.no_grad():
            z = self.autoencoder.encoder(x_stft)  # [B, S, D]
        
        return z
    
    def train_epoch(self) -> dict:
        """Run one training epoch."""
        self.model.train()
        losses = []
        
        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            # Get latents
            if self.use_cached_latents:
                z = batch.to(self.device, non_blocking=True)  # [B, S, D]
            else:
                z = self.encode_batch(batch)
            
            # Normalize latents to N(0,1) for diffusion
            z = self.normalize_latents(z)
            
            # Forward pass (diffusion loss)
            with autocast("cuda", enabled=self.use_amp):
                result = self.model(z)
                loss = result['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Learning rate warmup
            lr_scale = self.get_lr_scale()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * lr_scale
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update EMA
            if self.use_ema:
                self.ema.update(self.model)
            
            self.global_step += 1
            losses.append(loss.item())
        
        return {"loss": np.mean(losses)}
    
    def val_epoch(self) -> dict:
        """Run one validation epoch."""
        self.model.eval()
        losses = []
        
        # Use EMA weights for validation if available
        if self.use_ema:
            self.ema.apply_shadow(self.model)
        
        try:
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Val", leave=False):
                    # Get latents
                    if self.use_cached_latents:
                        z = batch.to(self.device, non_blocking=True)
                    else:
                        z = self.encode_batch(batch)
                    
                    # Normalize latents to N(0,1) for diffusion
                    z = self.normalize_latents(z)
                    
                    with autocast("cuda", enabled=self.use_amp):
                        result = self.model(z)
                        loss = result['loss']
                    
                    losses.append(loss.item())
        finally:
            # Restore original weights
            if self.use_ema:
                self.ema.restore(self.model)
        
        return {"loss": np.mean(losses)}
    
    @torch.no_grad()
    def generate_samples(self, epoch: int):
        """Generate and save audio samples."""
        self.model.eval()
        
        # Use EMA weights for sampling
        if self.use_ema:
            self.ema.apply_shadow(self.model)
        
        try:
            # Create output directory
            epoch_dir = os.path.join(self.samples_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            self.logs.info(f"Generating {self.num_samples} samples...")
            
            # Generate latents via diffusion (in normalized N(0,1) space)
            # Use DDIM sampling for stability
            z_samples = self.model.sample(
                batch_size=self.num_samples,
                device=self.device,
                show_progress=True,
                use_ddim=True,
                ddim_steps=50,
                eta=0.0,
            )  # [N, S, D]
            
            # Denormalize back to original latent scale
            z_samples = self.denormalize_latents(z_samples)
            
            # Decode to audio
            audio_samples = self.autoencoder.decoder(z_samples)  # [N, 1, L]
            
            # Save each sample
            for i in range(self.num_samples):
                audio = audio_samples[i, 0].cpu().numpy()  # [L]
                
                # Normalize to prevent clipping
                max_val = np.abs(audio).max()
                if max_val > 0:
                    audio = audio / max_val * 0.95
                
                path = os.path.join(epoch_dir, f"sample_{i:03d}.wav")
                sf.write(path, audio, self.sample_rate, subtype="FLOAT")
            
            self.logs.info(f"Samples saved to {epoch_dir}")
            
        finally:
            if self.use_ema:
                self.ema.restore(self.model)
    
    def save_checkpoint(self, epoch: int, val_loss: float, best_val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'diffusion_cfg': self.diffusion_cfg,
        }
        
        if self.use_ema:
            ckpt['ema_state_dict'] = self.ema.state_dict()
        
        # Save latent normalization statistics
        if self.z_mean is not None and self.z_std is not None:
            ckpt['z_mean'] = self.z_mean.cpu()
            ckpt['z_std'] = self.z_std.cpu()
        
        if is_best:
            path = os.path.join(self.save_path, "best_model.pth")
            torch.save(ckpt, path)
            self.logs.info(f"Best model saved: {path}")
        else:
            path = os.path.join(self.save_path, f"checkpoint_{epoch+1}.pth")
            torch.save(ckpt, path)
            self.logs.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> tuple:
        """Load checkpoint to resume training."""
        self.logs.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        if self.use_ema and 'ema_state_dict' in ckpt:
            self.ema.load_state_dict(ckpt['ema_state_dict'])
        
        # Load latent normalization statistics
        if 'z_mean' in ckpt and 'z_std' in ckpt:
            self.z_mean = ckpt['z_mean'].to(self.device)
            self.z_std = ckpt['z_std'].to(self.device)
            self.logs.info(f"Loaded latent statistics: mean avg={self.z_mean.mean().item():.4f}, std avg={self.z_std.mean().item():.4f}")
        
        self.global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        
        self.logs.info(f"Resumed from epoch {start_epoch}, step {self.global_step}, best val loss: {best_val_loss:.6f}")
        return start_epoch, best_val_loss
    
    def fit(self, start_epoch: int = 0, best_val_loss: float = float('inf')):
        """Main training loop."""
        for epoch in range(start_epoch, self.num_epochs):
            self.logs.info(f"=== Epoch {epoch+1}/{self.num_epochs} ===")
            
            # Update sampler epoch for deterministic shuffling (if using ShardedSampler)
            if hasattr(self, 'train_sampler') and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()
            
            # Log metrics
            lr = self.optimizer.param_groups[0]['lr']
            self.logs.info(
                f"Train Loss: {train_metrics['loss']:.6f}, "
                f"Val Loss: {val_metrics['loss']:.6f}, "
                f"LR: {lr:.2e}, "
                f"Step: {self.global_step}"
            )
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics['loss'], best_val_loss, is_best=True)
            
            # Periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch, val_metrics['loss'], best_val_loss)
            
            # Generate samples
            if (epoch + 1) % self.sample_interval == 0:
                self.generate_samples(epoch)
        
        self.logs.info(f"Training complete. Best val loss: {best_val_loss:.6f}")


def cache_latents(trainer: DiffusionTrainer, chunks_dir: str, output_dir: str):
    """
    Pre-compute and cache latent vectors for all training data.
    
    This speeds up training by avoiding encoder forward passes every step.
    Run this once before training.
    """
    import json
    
    trainer.logs.info("Caching latents...")
    trainer.autoencoder.eval()
    
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(chunks_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        output_split_dir = os.path.join(output_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)
        
        # Get files
        files = sorted([f for f in os.listdir(split_dir) if f.endswith('.pt')])
        
        trainer.logs.info(f"Processing {split}: {len(files)} files")
        
        for filename in tqdm(files, desc=f"Caching {split}"):
            input_path = os.path.join(split_dir, filename)
            output_path = os.path.join(output_split_dir, filename)
            
            # Skip if already cached
            if os.path.exists(output_path):
                continue
            
            # Load STFT (batched: multiple chunks per file)
            data = torch.load(input_path, weights_only=True)
            x_stft = data['x_stft'].to(trainer.device).float()  # [N, 2, F, T] - convert to float32
            
            # Encode in batches to avoid OOM
            batch_size = 32
            all_z = []
            for i in range(0, x_stft.size(0), batch_size):
                batch = x_stft[i:i+batch_size]
                with torch.no_grad():
                    z = trainer.autoencoder.encoder(batch)  # [B, S, D]
                all_z.append(z.cpu())
            
            # Concatenate and save
            z_all = torch.cat(all_z, dim=0)  # [N, S, D]
            torch.save({'z': z_all}, output_path)
    
    # Create index with chunk counts (same format as STFT index)
    index = {}
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            split_index = {}
            for filename in sorted(os.listdir(split_dir)):
                if filename.endswith('.pt'):
                    path = os.path.join(split_dir, filename)
                    data = torch.load(path, weights_only=True)
                    split_index[filename] = data['z'].size(0)  # Number of chunks
            index[split] = split_index
    
    with open(os.path.join(output_dir, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2)
    
    trainer.logs.info(f"Latents cached to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Latent Diffusion Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--cache-latents", action="store_true", 
                        help="Pre-compute and cache latents before training")
    parser.add_argument("--recompute-stats", action="store_true",
                        help="Force recomputation of latent statistics")
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DiffusionTrainer(args.config)
    
    # Load autoencoder
    ae_config = trainer.load_autoencoder()
    
    # Cache latents if requested
    if args.cache_latents:
        cache_latents(
            trainer,
            trainer.data_cfg['chunks_dir'],
            trainer.data_cfg['latents_cache_dir'],
        )
    
    # Check if cached latents exist
    latents_dir = trainer.data_cfg.get('latents_cache_dir')
    use_cached = (
        trainer.data_cfg.get('cache_latents', False) and 
        latents_dir and 
        os.path.exists(latents_dir)
    )
    
    if use_cached:
        trainer.logs.info(f"Using cached latents from {latents_dir}")
    else:
        trainer.logs.info("Encoding latents on-the-fly (consider --cache-latents for speed)")
    
    # Build dataloaders
    trainer.build_dataloaders(use_cached_latents=use_cached)
    
    # Build diffusion model
    trainer.build_model()
    
    # Resume or start fresh
    if args.resume:
        start_epoch, best_val_loss = trainer.load_checkpoint(args.resume)
        # If resuming and stats not in checkpoint, try loading from file or recompute
        if trainer.z_mean is None or trainer.z_std is None:
            if not trainer.load_latent_statistics():
                trainer.logs.info("No latent statistics found, computing...")
                trainer.compute_latent_statistics()
        trainer.fit(start_epoch=start_epoch, best_val_loss=best_val_loss)
    else:
        # For fresh training: compute or load latent statistics
        if args.recompute_stats or not trainer.load_latent_statistics():
            trainer.compute_latent_statistics()
        trainer.fit()

