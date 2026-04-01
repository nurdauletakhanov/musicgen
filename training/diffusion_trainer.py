"""Trainer for latent diffusion model (DiT on pre-extracted autoencoder latents).

Prerequisites:
    1. Train autoencoder (Phase 1 + Phase 2)
    2. Run data/extract_latents.py to cache latent tokens
    3. Run this trainer on cached latents
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.latent_dataset import LatentDataset
from models.diffusion import DiT, DDPMScheduler
from training.config import get_device, load_config
from training.utils import print_logger


class DiffusionTrainer:
    """Training orchestrator for latent DiT model."""

    def __init__(self, config_path, base_config_path=None):
        cfg = load_config(config_path, base_config_path)
        self.cfg = cfg
        diff_cfg = cfg['diffusion']
        train_cfg = cfg['train']

        # Device
        self.device = get_device(train_cfg.get('gpu_index', 0))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Save path
        name = cfg.get('name', 'diffusion-default')
        self.save_path = os.path.join(
            train_cfg.get('save_path', './checkpoints'), name
        )
        os.makedirs(self.save_path, exist_ok=True)
        self.logs = print_logger(self.save_path)
        self.logs.info(f"Device: {self.device}")

        # Training params
        self.batch_size = train_cfg['batch_size']
        self.num_epochs = train_cfg['num_epochs']
        self.learning_rate = train_cfg['learning_rate']
        self.weight_decay = train_cfg.get('weight_decay', 0.01)
        self.grad_clip = train_cfg.get('grad_clip', 1.0)
        self.save_interval = train_cfg.get('save_interval', 10)
        self.sample_interval = train_cfg.get('sample_interval', 20)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 5)
        self.num_workers = train_cfg.get('num_workers', 4)

        # AMP
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        # Diffusion params
        self.num_timesteps = diff_cfg.get('num_timesteps', 1000)
        self.scheduler = DDPMScheduler(self.num_timesteps)
        self.ddim_steps = diff_cfg.get('ddim_steps', 50)

        # Classifier-free guidance
        self.cfg_drop_prob = diff_cfg.get('cfg_drop_prob', 0.1)
        self.cfg_scale = diff_cfg.get('cfg_scale', 0.0)

        # Data paths
        self.latent_dir = diff_cfg['latent_dir']
        self.normalize_latents = diff_cfg.get('normalize_latents', True)

        # Autoencoder decoder for sample generation
        self.ae_checkpoint = diff_cfg.get('ae_checkpoint', None)

        # DiT model config
        self.dit_config = {
            'seq_len': diff_cfg.get('seq_len', 96),
            'd_model': diff_cfg.get('d_model', 96),
            'depth': diff_cfg.get('depth', 8),
            'n_heads': diff_cfg.get('n_heads', 6),
            'mlp_ratio': diff_cfg.get('mlp_ratio', 4.0),
            'num_classes': diff_cfg.get('num_classes', 4),
            'dropout': diff_cfg.get('dropout', 0.0),
        }

        self.sample_rate = cfg.get('data', {}).get('sample_rate', 44100)

    def build_model(self):
        """Build DiT, optimizer, LR scheduler."""
        self.model = DiT(**self.dit_config).to(self.device)
        self.logs.info(f"DiT params: {self.model.num_parameters():,}")
        self.logs.info(f"DiT config: {self.dit_config}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-3, end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs - self.warmup_epochs,
                eta_min=1e-6,
            )
            self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs, eta_min=1e-6,
            )

        self.scheduler.to(self.device)

    def prepare_data(self):
        """Load pre-extracted latent datasets."""
        train_ds = LatentDataset(self.latent_dir, 'train',
                                 normalize=self.normalize_latents)
        val_ds = LatentDataset(self.latent_dir, 'val',
                                normalize=self.normalize_latents)

        # Store for denormalization during sample generation
        self.train_dataset = train_ds

        self.train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        self.logs.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        # Load autoencoder decoder for sample generation
        self.ae_decoder = None
        if self.ae_checkpoint and os.path.exists(self.ae_checkpoint):
            from models.autoencoder import Autoencoder
            ckpt = torch.load(self.ae_checkpoint, map_location='cpu',
                              weights_only=False)
            ae = Autoencoder(**ckpt['model_config'])
            ae.load_state_dict(ckpt['model_state_dict'])
            self.ae_decoder = ae.decoder.to(self.device)
            self.ae_decoder.eval()
            del ae.encoder
            self.logs.info("Loaded autoencoder decoder for sample generation")

    def _compute_loss(self, z_0, y):
        """Diffusion training step: noise → predict → MSE."""
        B = z_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        noise = torch.randn_like(z_0)
        z_t = self.scheduler.add_noise(z_0, noise, t)

        # CFG: randomly drop class to unconditional token
        if self.cfg_drop_prob > 0 and self.model.num_classes > 0:
            drop = torch.rand(B, device=self.device) < self.cfg_drop_prob
            y = y.clone()
            y[drop] = self.model.num_classes  # unconditional index

        noise_pred = self.model(z_t, t, y)
        return F.mse_loss(noise_pred, noise)

    def _run_epoch(self, loader, training=True):
        """Run one epoch, return mean loss."""
        self.model.train() if training else self.model.eval()
        losses = []
        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for z_0, y in tqdm(loader, desc="Train" if training else "Val",
                               leave=False):
                z_0 = z_0.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with autocast("cuda", dtype=self.amp_dtype,
                              enabled=self.use_amp):
                    loss = self._compute_loss(z_0, y)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    self.optimizer.step()

                losses.append(loss.item())

        return np.mean(losses)

    @torch.no_grad()
    def generate_samples(self, num_samples=4, stem_class=None):
        """Generate latents via DDIM, decode to audio."""
        self.model.eval()
        shape = (num_samples, self.dit_config['seq_len'],
                 self.dit_config['d_model'])

        y = None
        if stem_class is not None and self.model.num_classes > 0:
            y = torch.full((num_samples,), stem_class, device=self.device,
                           dtype=torch.long)

        z_0 = self.scheduler.sample_ddim(
            self.model, shape, self.device,
            num_steps=self.ddim_steps, y=y, progress=True,
        )

        # Denormalize back to autoencoder scale
        z_0 = self.train_dataset.denormalize(z_0)

        if self.ae_decoder is None:
            return z_0  # return raw latents if no decoder loaded

        waveforms = []
        for i in range(num_samples):
            with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                wav, _ = self.ae_decoder(z_0[i:i+1])
            waveforms.append(wav.cpu().squeeze(0))
        return waveforms

    def _save_samples(self, epoch):
        """Generate and save audio samples for each stem type."""
        if self.ae_decoder is None:
            return

        import torchaudio
        from data.latent_dataset import STEM_CLASSES

        sample_dir = os.path.join(self.save_path, 'samples', f'epoch_{epoch}')
        os.makedirs(sample_dir, exist_ok=True)

        for stem_name, stem_idx in STEM_CLASSES.items():
            wavs = self.generate_samples(num_samples=2, stem_class=stem_idx)
            for i, wav in enumerate(wavs):
                path = os.path.join(sample_dir, f'{stem_name}_{i}.wav')
                torchaudio.save(path, wav, self.sample_rate)

        self.logs.info(f"Saved samples to {sample_dir}")

    def fit(self, start_epoch=0, best_val_loss=float('inf')):
        """Main training loop."""
        self.logs.info(f"Training: {self.num_epochs} epochs, "
                       f"batch_size={self.batch_size}, lr={self.learning_rate}")

        for epoch in range(start_epoch, self.num_epochs):
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)
            self.lr_scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            self.logs.info(
                f"Epoch {epoch+1}/{self.num_epochs} — "
                f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {lr:.2e}"
            )

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'dit_config': self.dit_config,
                }, os.path.join(self.save_path, 'best_model.pth'))
                self.logs.info(f"Best val loss: {val_loss:.6f}")

            # Periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'dit_config': self.dit_config,
                }, os.path.join(self.save_path, f'checkpoint_epoch{epoch+1}.pth'))

            # Generate samples
            if (epoch + 1) % self.sample_interval == 0:
                self._save_samples(epoch + 1)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.logs.info(f"Done. Best val loss: {best_val_loss:.6f}")
