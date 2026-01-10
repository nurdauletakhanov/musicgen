import os
import argparse
import random

import torch
import torch.nn as nn
import numpy as np
import yaml
import soundfile as sf
from tqdm import tqdm
from torch.nn import functional as F
from torch.amp import autocast, GradScaler

from models.autoencoder import Autoencoder
from training.utils import print_logger
from data.dataloader import build_dataloaders


def get_device(gpu_index: int = 0) -> torch.device:
    """Get CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        if gpu_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested GPU {gpu_index} but only {torch.cuda.device_count()} available."
            )
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


class Trainer:
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.cfg = cfg
        train_cfg = cfg['train']
        model_cfg = cfg['model']
        data_cfg = cfg['data']
        stft_cfg = cfg.get('stft', {})
        
        # Device setup
        self.device = get_device(train_cfg.get('gpu_index', 0))
        
        # Training params
        self.batch_size = train_cfg['batch_size']
        self.num_epochs = train_cfg['num_epochs']
        self.learning_rate = train_cfg['learning_rate']
        self.weight_decay = train_cfg['weight_decay']
        self.patience = train_cfg.get('patience', 5)
        self.factor = train_cfg.get('factor', 0.5)
        self.save_interval = train_cfg.get('save_interval', 10)
        self.num_workers = train_cfg.get('num_workers', 4)  
        self.pin_memory = train_cfg.get('pin_memory', None)
        self.persistent_workers = train_cfg.get('persistent_workers', None)
        self.prefetch_factor = train_cfg.get('prefetch_factor', 2)  # Default 2 for prefetching
        self.reset_scheduler_on_resume = train_cfg.get('reset_scheduler_on_resume', False)
        self.scheduler_warmup_epochs = train_cfg.get('scheduler_warmup_epochs', 0)
        
        # Save path
        name = cfg.get('name', 'default')
        base_path = train_cfg.get('save_path', './checkpoints')
        self.save_path = os.path.join(base_path, name)
        os.makedirs(self.save_path, exist_ok=True)
        
        # Logger
        self.logs = print_logger(self.save_path)
        self.logs.info(f"Device: {self.device}")
        if self.device.type == "cuda":
            self.logs.info(f"GPU: {torch.cuda.get_device_name(self.device)}")

        # Compute STFT-derived model params
        n_fft = stft_cfg.get('n_fft', 1024)
        win_length = stft_cfg.get('win_length', n_fft)
        hop_length = stft_cfg.get('hop_length', 256)
        sample_rate = data_cfg.get('sample_rate', 44100)
        chunk_seconds = data_cfg.get('chunk_seconds', 1.0)
        
        n_freq_bins = n_fft // 2 + 1
        chunk_samples = int(sample_rate * chunk_seconds)
        target_length = chunk_samples  # Target waveform length in samples
        
        # Store for audio saving
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.num_test_samples = train_cfg.get('num_test_samples', 5)  # Number of samples to save
        
        # Require upsampling_factors to be manually specified
        num_segments = model_cfg['num_segments']
        channels = model_cfg['channels']
        num_upsample_blocks = len(channels) - 1
        
        if 'upsampling_factors' not in model_cfg:
            raise ValueError(
                "upsampling_factors must be specified in config. "
                f"Expected a list of {num_upsample_blocks} factors for {len(channels)} channels."
            )
        
        upsampling_factors = model_cfg['upsampling_factors']
        if len(upsampling_factors) != num_upsample_blocks:
            raise ValueError(
                f"upsampling_factors length ({len(upsampling_factors)}) must match "
                f"number of upsampling blocks ({num_upsample_blocks})"
            )
        
        # Model config with computed values
        self.model_config = {
            'd_model': model_cfg['d_model'],
            'n_heads': model_cfg['n_heads'],
            'n_layers': model_cfg['n_layers'],
            'num_segments': num_segments,
            'n_freq_bins': n_freq_bins,
            'channels': channels,
            'upsampling_factors': upsampling_factors,
            'target_length': target_length,
            'latent_mix_weight': model_cfg.get('latent_mix_weight', 0.0),
            'decode_mix_weight': model_cfg.get('decode_mix_weight', 0.0),
            'mrstft_weight': model_cfg.get('mrstft_weight', 1.0),
            'l1_weight': model_cfg.get('l1_weight', 1.0),
            'mix_l1_weight': model_cfg.get('mix_l1_weight', 1.0),
            'mix_mrstft_weight': model_cfg.get('mix_mrstft_weight', 1.0),
            'dropout': model_cfg.get('dropout', 0.1),
            # STFT params for on-the-fly computation in decode mixing loss
            'n_fft': n_fft,
            'hop_length': hop_length,
            'win_length': win_length,
        }

        # Data loaders with sharded sampler (sequential file access to avoid disk thrashing)
        (
            self.train_loader,
            self.val_loader,
            self.train_dataset,
            self.val_dataset,
            self.train_sampler,
        ) = build_dataloaders(
            chunks_dir=data_cfg['chunks_dir'],
            train_split=data_cfg.get('split', 'train'),
            val_split=data_cfg.get('val_split', 'validation'),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            latent_mix_weight=self.model_config['latent_mix_weight'],
            decode_mix_weight=self.model_config['decode_mix_weight'],
            logger=self.logs,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )
        
        # Log data loading configuration for debugging GPU utilization
        self.logs.info(f"Data loading: num_workers={self.num_workers}, prefetch_factor={self.prefetch_factor}, "
                      f"pin_memory={self.pin_memory}, persistent_workers={self.persistent_workers}")
        
        # AMP setup
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.logs.info(f"Mixed precision (AMP): {'enabled' if self.use_amp else 'disabled'}")

    def build_model(self):
        """Build model, optimizer, and scheduler."""
        self.logs.info(f"Model config: {self.model_config}")
        self.model = Autoencoder(**self.model_config).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=self.patience, 
            factor=self.factor,
            threshold=1e-3,
            threshold_mode='rel',
            cooldown=1,
            min_lr=1e-6,
        )

        # Log parameters
        total = sum(p.numel() for p in self.model.parameters())
        encoder_params = self.model.encoder.num_parameters()
        decoder_params = self.model.decoder.num_parameters()
        
        # encoder.num_parameters() returns a dict, extract total
        if isinstance(encoder_params, dict):
            encoder_total = encoder_params.get('total', encoder_params)
        else:
            encoder_total = encoder_params
        
        self.logs.info(f"=== Parameters ===")
        self.logs.info(f"Total: {total:,}")
        self.logs.info(f"Encoder: {encoder_total:,}")
        self.logs.info(f"Decoder: {decoder_params:,}")

    def load_checkpoint(self, checkpoint_path: str) -> tuple:
        """Load checkpoint to resume training. Returns (start_epoch, best_val_loss)."""
        self.logs.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Use strict=False to allow loading older checkpoints missing new buffers (e.g. stft_window)
        missing, unexpected = self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing:
            self.logs.info(f"Missing keys (using defaults): {missing}")
        if unexpected:
            self.logs.info(f"Unexpected keys (ignored): {unexpected}")
        
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # Handle scheduler state based on config
        if self.reset_scheduler_on_resume:
            self.logs.info("Resetting scheduler state (loss landscape may have changed)")
            # Recreate scheduler with fresh state
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',
                patience=self.patience, 
                factor=self.factor,
                threshold=1e-3,
                threshold_mode='rel',
                cooldown=1,
                min_lr=1e-6,
            )
        else:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if self.scheduler_warmup_epochs > 0:
                self.logs.info(f"Scheduler warmup enabled: LR reduction disabled for {self.scheduler_warmup_epochs} epochs after resume")
        
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        
        self.logs.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
        return start_epoch, best_val_loss

    def train_epoch(self) -> dict:
        """Run one training epoch."""
        self.model.train()
        losses = []
        recon_losses = []
        wav_l1_losses = []
        mrstft_losses = []
        latent_mix_losses = []
        decode_mix_losses = []
        decode_mix_l1_losses = []
        decode_mix_mrstft_losses = []
        decode_mix_rates = []

        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            # Extract batch data (dataloader returns dict with x_stft and x_wave)
            x_stft = batch['x_stft'].to(self.device, non_blocking=True)  # [B, 2, F, T]
            x_wave = batch['x_wave'].to(self.device, non_blocking=True)  # [B, L]
            
            # Reshape x_wave to [B, 1, L] as expected by model
            if x_wave.dim() == 2:
                x_wave = x_wave.unsqueeze(1)  # [B, 1, L]
            
            with autocast("cuda", enabled=self.use_amp):
                total, components = self.model(x_stft, x_wave)

            self.optimizer.zero_grad()
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(total.item())
            recon_losses.append(components.get('recon', 0.0))
            wav_l1_losses.append(components.get('wav_l1', 0.0))
            mrstft_losses.append(components.get('mrstft', 0.0))
            latent_mix_losses.append(components.get('latent_mix', 0.0))
            decode_mix_losses.append(components.get('decode_mix', 0.0))
            decode_mix_l1_losses.append(components.get('decode_mix_l1', 0.0))
            decode_mix_mrstft_losses.append(components.get('decode_mix_mrstft', 0.0))
            decode_mix_rates.append(components.get('rate', 0.0))

        return {
            "loss": np.mean(losses),
            "recon": np.mean(recon_losses),
            "wav_l1": np.mean(wav_l1_losses),
            "mrstft": np.mean(mrstft_losses),
            "latent_mix": np.mean(latent_mix_losses),
            "decode_mix": np.mean(decode_mix_losses),
            "decode_mix_l1": np.mean(decode_mix_l1_losses),
            "decode_mix_mrstft": np.mean(decode_mix_mrstft_losses),
            "decode_mix_rate": np.mean(decode_mix_rates),
        }

    def val_epoch(self) -> dict:
        """Run one validation epoch."""
        self.model.eval()
        losses = []
        recon_losses = []
        wav_l1_losses = []
        mrstft_losses = []
        latent_mix_losses = []
        decode_mix_losses = []
        decode_mix_l1_losses = []
        decode_mix_mrstft_losses = []
        decode_mix_rates = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val", leave=False):
                # Extract batch data (dataloader returns dict with x_stft and x_wave)
                x_stft = batch['x_stft'].to(self.device, non_blocking=True)  # [B, 2, F, T]
                x_wave = batch['x_wave'].to(self.device, non_blocking=True)  # [B, L]
                
                # Reshape x_wave to [B, 1, L] as expected by model
                if x_wave.dim() == 2:
                    x_wave = x_wave.unsqueeze(1)  # [B, 1, L]
                
                with autocast("cuda", enabled=self.use_amp):
                    total, components = self.model(x_stft, x_wave)
                    
                losses.append(total.item())
                recon_losses.append(components.get('recon', 0.0))
                wav_l1_losses.append(components.get('wav_l1', 0.0))
                mrstft_losses.append(components.get('mrstft', 0.0))
                latent_mix_losses.append(components.get('latent_mix', 0.0))
                decode_mix_losses.append(components.get('decode_mix', 0.0))
                decode_mix_l1_losses.append(components.get('decode_mix_l1', 0.0))
                decode_mix_mrstft_losses.append(components.get('decode_mix_mrstft', 0.0))
                decode_mix_rates.append(components.get('rate', 0.0))

        return {
            "loss": np.mean(losses),
            "recon": np.mean(recon_losses),
            "wav_l1": np.mean(wav_l1_losses),
            "mrstft": np.mean(mrstft_losses),
            "latent_mix": np.mean(latent_mix_losses),
            "decode_mix": np.mean(decode_mix_losses),
            "decode_mix_l1": np.mean(decode_mix_l1_losses),
            "decode_mix_mrstft": np.mean(decode_mix_mrstft_losses),
            "decode_mix_rate": np.mean(decode_mix_rates),
        }

    def save_test_samples(self, epoch: int):
        """Save original and reconstructed audio samples for testing."""
        self.model.eval()
        
        # Create samples directory
        samples_dir = os.path.join(self.save_path, "samples", f"epoch_{epoch+1}")
        os.makedirs(samples_dir, exist_ok=True)
        
        # Get random samples from validation set
        num_samples = min(self.num_test_samples, len(self.val_dataset))
        sample_indices = random.sample(range(len(self.val_dataset)), num_samples)
        
        self.logs.info(f"Saving {num_samples} test samples to {samples_dir}")
        
        with torch.no_grad():
            for i, idx in enumerate(sample_indices):
                sample = self.val_dataset[idx]
                x_stft = sample["x_stft"].unsqueeze(0).to(self.device)  # [1, 2, F, T]
                x_wave = sample["x_wave"].unsqueeze(0).to(self.device)  # [1, L]
                
                # Ensure x_wave has channel dimension [1, 1, L]
                if x_wave.dim() == 2:
                    x_wave = x_wave.unsqueeze(1)  # [1, 1, L]
                
                # Reconstruct
                with autocast("cuda", enabled=self.use_amp):
                    z = self.model.encoder(x_stft)  # [1, S, D]
                    y_hat = self.model.decoder(z)   # [1, 1, target_length]
                
                # Convert to CPU and remove batch/channel dims
                x_wave_orig = x_wave[0, 0].cpu().float()  # [L]
                y_hat_recon = y_hat[0, 0].cpu().float()   # [target_length]
                
                # Ensure lengths match chunk_samples (trim or pad)
                if x_wave_orig.numel() > self.chunk_samples:
                    x_wave_orig = x_wave_orig[:self.chunk_samples]
                elif x_wave_orig.numel() < self.chunk_samples:
                    x_wave_orig = F.pad(x_wave_orig, (0, self.chunk_samples - x_wave_orig.numel()), value=0.0)
                
                if y_hat_recon.numel() > self.chunk_samples:
                    y_hat_recon = y_hat_recon[:self.chunk_samples]
                elif y_hat_recon.numel() < self.chunk_samples:
                    y_hat_recon = F.pad(y_hat_recon, (0, self.chunk_samples - y_hat_recon.numel()), value=0.0)
                
                # Save as WAV files
                orig_path = os.path.join(samples_dir, f"sample_{i:03d}_original.wav")
                recon_path = os.path.join(samples_dir, f"sample_{i:03d}_reconstructed.wav")
                
                # Convert to numpy and save
                audio_orig = x_wave_orig.numpy()
                audio_recon = y_hat_recon.numpy()
                
                sf.write(orig_path, audio_orig, self.sample_rate, subtype="FLOAT")
                sf.write(recon_path, audio_recon, self.sample_rate, subtype="FLOAT")
        
        self.logs.info(f"Test samples saved to {samples_dir}")

    def save_checkpoint(self, epoch: int, val_loss: float, best_val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            'epoch': epoch,
            'model_config': self.model_config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }
        
        if is_best:
            path = os.path.join(self.save_path, "best_model.pth")
            torch.save(ckpt, path)
            self.logs.info(f"Best model saved: {path}")
        else:
            path = os.path.join(self.save_path, f"checkpoint_{epoch+1}.pth")
            torch.save(ckpt, path)
            self.logs.info(f"Checkpoint saved: {path}")

    def fit(self, start_epoch: int = 0, best_val_loss: float = float('inf')):
        """Main training loop."""
        # Calculate warmup end epoch if resuming with warmup enabled
        scheduler_warmup_end = start_epoch + self.scheduler_warmup_epochs if start_epoch > 0 else 0
        
        for epoch in range(start_epoch, self.num_epochs):
            self.logs.info(f"=== Epoch {epoch+1}/{self.num_epochs} ===")
            
            # Update sampler epoch for deterministic shuffling
            self.train_sampler.set_epoch(epoch)
            
            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()
            
            # Update scheduler (skip during warmup period after resume)
            if epoch < scheduler_warmup_end:
                self.logs.info(f"Scheduler warmup: skipping LR step (epoch {epoch+1}/{scheduler_warmup_end})")
            else:
                self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self.logs.info(
                f"Train - Loss: {train_metrics['loss']:.6f}, "
                f"Recon: {train_metrics['recon']:.6f}, "
                f"WavL1: {train_metrics['wav_l1']:.6f}, "
                f"MRSTFT: {train_metrics['mrstft']:.6f}, "
                f"LatMix: {train_metrics['latent_mix']:.6f}, "
                f"DecMix: {train_metrics['decode_mix']:.6f}, "
                f"DecMixL1: {train_metrics['decode_mix_l1']:.6f}, "
                f"DecMixMRSTFT: {train_metrics['decode_mix_mrstft']:.6f}, "
                f"DecMixRate: {train_metrics['decode_mix_rate']:.4f}"
            )
            self.logs.info(
                f"Val   - Loss: {val_metrics['loss']:.6f}, "
                f"Recon: {val_metrics['recon']:.6f}, "
                f"WavL1: {val_metrics['wav_l1']:.6f}, "
                f"MRSTFT: {val_metrics['mrstft']:.6f}, "
                f"LatMix: {val_metrics['latent_mix']:.6f}, "
                f"DecMix: {val_metrics['decode_mix']:.6f}, "
                f"DecMixL1: {val_metrics['decode_mix_l1']:.6f}, "
                f"DecMixMRSTFT: {val_metrics['decode_mix_mrstft']:.6f}, "
                f"DecMixRate: {val_metrics['decode_mix_rate']:.4f}"
            )
            self.logs.info(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics['loss'], best_val_loss, is_best=True)
                # Save test samples when best model is updated
                self.save_test_samples(epoch)
            
            # Periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch, val_metrics['loss'], best_val_loss)

        self.logs.info(f"Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.build_model()
    
    if args.resume:
        start_epoch, best_val_loss = trainer.load_checkpoint(args.resume)
        trainer.fit(start_epoch=start_epoch, best_val_loss=best_val_loss)
    else:
        trainer.fit()
