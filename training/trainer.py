"""Core Trainer class for STFT Autoencoder training."""

import os

import numpy as np
import torch
from tqdm import tqdm

from data.dataloader import build_single_stem_dataloaders
from evaluation.samples import save_test_samples
from evaluation.utils import si_sdr
from models.autoencoder import Autoencoder
from models.discriminator import (
    MultiScaleSTFTDiscriminator,
    discriminator_loss,
    generator_loss,
    feature_matching_loss,
)
from training.checkpoint import save_checkpoint, load_checkpoint
from training.config import get_device, load_config, build_model_config
from training.hub_utils import setup_hub_integration
from training.tb_logger import TBLogger, get_alpha_sweep_epochs
from training.utils import print_logger


class Trainer:
    """Training orchestrator for STFT Autoencoder models."""

    def __init__(self, config_path: str, base_config_path: str = None):
        cfg = load_config(config_path, base_config_path)
        
        self.cfg = cfg
        self.config_path = config_path
        train_cfg = cfg['train']
        data_cfg = cfg['data']
        hf_cfg = cfg.get('huggingface', {})
        
        # Device setup
        self.device = get_device(train_cfg.get('gpu_index', 0))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Training params
        self.batch_size = train_cfg['batch_size']
        self.num_epochs = train_cfg['num_epochs']
        self.learning_rate = train_cfg['learning_rate']
        self.weight_decay = train_cfg['weight_decay']
        self.save_interval = train_cfg.get('save_interval', 10)
        self.num_workers = train_cfg.get('num_workers', 4)  
        self.pin_memory = train_cfg.get('pin_memory', None)
        self.persistent_workers = train_cfg.get('persistent_workers', None)
        self.prefetch_factor = train_cfg.get('prefetch_factor', 2)
        self.reset_scheduler_on_resume = train_cfg.get('reset_scheduler_on_resume', False)
        self.scheduler_warmup_epochs = train_cfg.get('scheduler_warmup_epochs', 0)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 0)
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

        # Audio params for sample saving
        self.sample_rate = data_cfg.get('sample_rate', 44100)
        chunk_seconds = data_cfg.get('chunk_seconds', 1.0)
        self.chunk_samples = int(self.sample_rate * chunk_seconds)
        self.num_test_samples = train_cfg.get('num_test_samples', 5)
        
        # Build model config
        self.model_config = build_model_config(cfg)

        # Data loaders (single stems)
        chunks_dir = data_cfg.get('chunks_dir') or data_cfg.get('musdb_chunks_dir')
        (
            self.train_loader,
            self.val_loader,
            self.train_dataset,
            self.val_dataset,
            self.train_sampler,
        ) = build_single_stem_dataloaders(
            chunks_dir=chunks_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            logger=self.logs,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

        self.logs.info(f"Batch size: {self.batch_size}")
        self.logs.info(
            f"Data loading: num_workers={self.num_workers}, "
            f"prefetch_factor={self.prefetch_factor}, "
            f"pin_memory={self.pin_memory}, "
            f"persistent_workers={self.persistent_workers}"
        )

        # Discriminator config (DAC-style MSSTFTD; off by default)
        model_cfg = cfg['model']
        self.use_discriminator = model_cfg.get('use_discriminator', False)
        self.disc_weight = model_cfg.get('disc_weight', 1.0)
        self.feat_match_weight = model_cfg.get('feat_match_weight', 2.0)
        self.disc_start_epoch = model_cfg.get('disc_start_epoch', 5)
        self.disc_channels = model_cfg.get('disc_channels', 32)
        self.disc_lr = model_cfg.get('disc_lr', 2e-4)

        # HuggingFace Hub setup
        hub = setup_hub_integration(
            hf_cfg, cfg.get('name', 'default'), config_path, logger=self.logs
        )
        self.hf_enabled = hub['enabled']
        self.hf_repo_id = hub['repo_id']
        self.hf_push_best = hub['push_best']
        self.hf_push_checkpoints = hub['push_checkpoints']
        self.hf_push_interval = hub['push_interval']
        self.hf_private = hub['private']
        
        # TensorBoard setup
        tb_cfg = cfg.get('tensorboard', {})
        self.tb_enabled = tb_cfg.get('enabled', True)
        self.tb_log_audio = tb_cfg.get('log_audio', True)
        self.tb_alpha_sweep_alphas = tb_cfg.get('alpha_sweep_alphas', [0.1, 0.3, 0.5, 0.7, 0.9])
        self.tb_alpha_sweep_samples = tb_cfg.get('alpha_sweep_samples', 100)
        self.tb_logger = TBLogger(self.save_path, enabled=self.tb_enabled)
        
        if self.tb_enabled:
            self.logs.info("TensorBoard logging enabled")

    def build_model(self):
        """Build model, optimizer, scheduler, and optional discriminator."""
        self.logs.info(f"Model config: {self.model_config}")
        self.model = Autoencoder(**self.model_config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs - self.warmup_epochs,
                eta_min=1e-6,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6,
            )

        # Discriminator (optional) — DAC-style MSSTFTD on complex STFT
        self.discriminator = None
        self.disc_optimizer = None
        if self.use_discriminator:
            self.discriminator = MultiScaleSTFTDiscriminator(
                channels=self.disc_channels,
            ).to(self.device)
            self.disc_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.disc_lr,
                betas=(0.5, 0.9),
                weight_decay=0.0,
            )

        # Log parameters
        total = sum(p.numel() for p in self.model.parameters())
        encoder_params = self.model.encoder.num_parameters()
        decoder_params = self.model.decoder.num_parameters()

        if isinstance(encoder_params, dict):
            encoder_total = encoder_params.get('total', encoder_params)
        else:
            encoder_total = encoder_params

        self.logs.info("=== Parameters ===")
        self.logs.info(f"Total: {total:,}")
        self.logs.info(f"Encoder: {encoder_total:,}")
        self.logs.info(f"Decoder: {decoder_params:,}")
        if self.discriminator is not None:
            disc_params = self.discriminator.num_parameters()
            self.logs.info(f"Discriminator: {disc_params:,}")
            self.logs.info(f"Disc starts at epoch {self.disc_start_epoch}")

    def load_checkpoint(self, checkpoint_path: str) -> tuple:
        """Load checkpoint to resume training."""
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            reset_scheduler=self.reset_scheduler_on_resume,
            logger=self.logs,
            discriminator=self.discriminator,
            disc_optimizer=self.disc_optimizer,
        )
        # Recreate cosine scheduler with correct T_max for remaining epochs
        if self.reset_scheduler_on_resume:
            remaining = self.num_epochs - start_epoch
            if self.warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=self.warmup_epochs,
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=remaining - self.warmup_epochs,
                    eta_min=1e-6,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[self.warmup_epochs],
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=remaining,
                    eta_min=1e-6,
                )
            self.logs.info(
                f"Cosine scheduler: T_max={remaining - self.warmup_epochs}, "
                f"warmup={self.warmup_epochs} epochs"
            )
        if self.scheduler_warmup_epochs > 0:
            self.logs.info(
                f"Scheduler warmup enabled: LR reduction disabled for "
                f"{self.scheduler_warmup_epochs} epochs after resume"
            )
        return start_epoch, best_val_loss

    def _run_epoch(self, loader, training: bool = True, epoch: int = 0) -> dict:
        """
        Run one epoch of training or validation.

        Args:
            loader: Primary data loader
            training: Whether this is a training epoch
            epoch: Current epoch number (for discriminator warmup)

        Returns dict with mean values for all metrics. MixRate is computed
        during validation only (requires a no_grad oracle pass per batch).
        """
        if training:
            self.model.train()
            if self.discriminator is not None:
                self.discriminator.train()
        else:
            self.model.eval()
            if self.discriminator is not None:
                self.discriminator.eval()

        disc_active = (self.discriminator is not None and epoch >= self.disc_start_epoch)

        # Metric collectors
        metric_keys = [
            'loss',
            'ReconSingle/Total', 'ReconSingle/MelL1', 'ReconSingle/MRSTFTMag',
            'DecodeMix/L1',
            'MixRecon/Total', 'MixRecon/MelL1', 'MixRecon/MRSTFTMag',
            'MixRate',  # val only
            'Latent/mean', 'Latent/std', 'Latent/absmax', 'Latent/l2',
            'grad_norm',
            'Disc/loss', 'Disc/gen_adv', 'Disc/feat_match',
            'SI-SDR',
            'Diag/RMSRatio', 'Diag/PhaseErr_deg',
        ]
        metrics = {k: [] for k in metric_keys}

        # Collect raw MixRate values for distribution stats (val only)
        mix_rate_values = []

        desc = "Train" if training else "Val"
        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
                x_stft = batch['x_stft'].to(self.device, non_blocking=True)
                x_wave = batch['x_wave'].to(self.device, non_blocking=True)

                if x_wave.dim() == 2:
                    x_wave = x_wave.unsqueeze(1)

                # --- Discriminator step (train only, after warmup) ---
                if training and disc_active:
                    self.discriminator.requires_grad_(True)
                    self.disc_optimizer.zero_grad()

                    with torch.no_grad():
                        z = self.model.encoder(x_wave)
                        x_hat = self.model.decoder(z)

                    tgt = self.model.decoder.target_length
                    x_real = x_wave[:, :, :tgt] if x_wave.size(2) > tgt else x_wave
                    real_logits, _ = self.discriminator(x_real)
                    fake_logits, _ = self.discriminator(x_hat.detach())
                    disc_loss = discriminator_loss(real_logits, fake_logits)

                    disc_loss.backward()
                    self.disc_optimizer.step()
                    metrics['Disc/loss'].append(disc_loss.item())
                    del disc_loss, real_logits, fake_logits, x_hat, z

                    self.discriminator.requires_grad_(False)

                # --- Generator step ---
                total, components, x_hat, _ = self.model(
                    x_stft, x_wave, compute_mix_rate=(not training)
                )

                # Adversarial + feature matching loss (reuse x_hat from forward)
                if disc_active and training:
                    tgt = self.model.decoder.target_length
                    x_real = x_wave[:, :, :tgt] if x_wave.size(2) > tgt else x_wave
                    fake_logits, fake_features = self.discriminator(x_hat)
                    with torch.no_grad():
                        _, real_features = self.discriminator(x_real)

                    gen_adv = generator_loss(fake_logits)
                    feat_match = feature_matching_loss(real_features, fake_features)
                    total = total + self.disc_weight * gen_adv + self.feat_match_weight * feat_match

                    components['Disc/gen_adv'] = gen_adv.item()
                    components['Disc/feat_match'] = feat_match.item()

                if training:
                    total.backward()

                if training:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    metrics['grad_norm'].append(grad_norm.item())
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Audio quality diagnostics (validation only)
                if not training and x_hat is not None:
                    tgt = self.model.decoder.target_length
                    x_ref = x_wave[:, :, :tgt] if x_wave.size(2) > tgt else x_wave
                    sisdr_val = si_sdr(x_hat.detach().float(), x_ref.float())
                    metrics['SI-SDR'].append(sisdr_val)

                    with torch.no_grad():
                        hat_f = x_hat.detach().float()
                        ref_f = x_ref.float()

                        # RMS ratio: energy preservation (1.0 = perfect)
                        rms_hat = hat_f.pow(2).mean(dim=-1).sqrt()
                        rms_ref = ref_f.pow(2).mean(dim=-1).sqrt()
                        metrics['Diag/RMSRatio'].append((rms_hat / rms_ref.clamp(min=1e-8)).mean().item())

                        # Phase error (magnitude-weighted, in degrees)
                        hat_stft = torch.stft(
                            hat_f.squeeze(1), n_fft=self.model.n_fft,
                            hop_length=self.model.hop_length,
                            win_length=self.model.win_length,
                            window=self.model.stft_window,
                            center=True, return_complex=True,
                        )
                        ref_stft = torch.stft(
                            ref_f.squeeze(1), n_fft=self.model.n_fft,
                            hop_length=self.model.hop_length,
                            win_length=self.model.win_length,
                            window=self.model.stft_window,
                            center=True, return_complex=True,
                        )
                        phase_diff = torch.abs(hat_stft.angle() - ref_stft.angle())
                        phase_diff = torch.min(phase_diff, 2 * 3.14159265 - phase_diff)
                        ref_mag = ref_stft.abs()
                        metrics['Diag/PhaseErr_deg'].append(
                            float((phase_diff * ref_mag).sum() / ref_mag.sum().clamp(min=1e-8) * 180 / 3.14159265)
                        )

                # Collect metrics
                metrics['loss'].append(total.item())
                for key in metrics.keys():
                    if key != 'loss' and key in components:
                        metrics[key].append(components[key])

                # Collect raw MixRate for distribution stats
                mix_rate = components.get('MixRate', 0.0)
                if mix_rate > 0:  # Only collect non-zero rates (mixing enabled)
                    mix_rate_values.append(mix_rate)

        result = {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
        result['_mix_rate_values'] = mix_rate_values  # Raw values for distribution
        return result

    def _save_checkpoint(self, epoch: int, val_loss: float, best_val_loss: float, is_best: bool = False):
        """Save checkpoint wrapper."""
        save_checkpoint(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            model_config=self.model_config,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            save_path=self.save_path,
            is_best=is_best,
            logger=self.logs,
            hf_enabled=self.hf_enabled,
            hf_repo_id=self.hf_repo_id,
            hf_push_best=self.hf_push_best,
            hf_push_checkpoints=self.hf_push_checkpoints,
            hf_push_interval=self.hf_push_interval,
            hf_private=self.hf_private,
            discriminator=self.discriminator,
            disc_optimizer=self.disc_optimizer,
        )

    def _log_metrics(self, prefix: str, metrics: dict):
        """Log training/validation metrics to console."""
        recon_total = metrics.get('ReconSingle/Total', 0.0)
        mel_l1 = metrics.get('ReconSingle/MelL1', 0.0)
        mrstft_mag = metrics.get('ReconSingle/MRSTFTMag', 0.0)
        decode_mix = metrics.get('DecodeMix/L1', 0.0)
        mix_recon = metrics.get('MixRecon/Total', 0.0)
        mix_mel = metrics.get('MixRecon/MelL1', 0.0)
        mix_rate = metrics.get('MixRate', 0.0)

        line = (
            f"{prefix} - Loss: {metrics['loss']:.6f} | "
            f"Recon: {recon_total:.4f} (Mel: {mel_l1:.4f}, MRSTFT: {mrstft_mag:.4f}) | "
            f"DecodeMix: {decode_mix:.4f} | "
            f"MixRecon: {mix_recon:.4f} (Mel: {mix_mel:.4f}) | "
            f"Rate: {mix_rate:.4f}"
        )

        sisdr = metrics.get('SI-SDR', 0.0)
        if sisdr != 0.0:
            line += f" | SI-SDR: {sisdr:.2f} dB"

        rms_ratio = metrics.get('Diag/RMSRatio', 0.0)
        if rms_ratio > 0.0:
            phase_err = metrics.get('Diag/PhaseErr_deg', 0.0)
            line += f" | RMS: {rms_ratio:.3f}, Phase: {phase_err:.1f}deg"

        disc_loss = metrics.get('Disc/loss', 0.0)
        if disc_loss > 0:
            gen_adv = metrics.get('Disc/gen_adv', 0.0)
            feat_match = metrics.get('Disc/feat_match', 0.0)
            line += f" | Disc: {disc_loss:.4f}, Adv: {gen_adv:.4f}, FM: {feat_match:.4f}"

        self.logs.info(line)

    def fit(self, start_epoch: int = 0, best_val_loss: float = float('inf')):
        """Main training loop."""
        scheduler_warmup_end = start_epoch + self.scheduler_warmup_epochs if start_epoch > 0 else 0
        
        # Log hyperparameters once at start
        self.tb_logger.log_training_config(self.cfg, self.model_config)
        
        # In-trainer alpha sweep is disabled: evaluation/alpha_sweep.py depends
        # on stale v6-era APIs (model.mrstft_loss, stem-pair test loaders). The
        # paper's MixRate numbers come from offline evaluation/sweep_evaluation.py.
        alpha_sweep_epochs = set()

        for epoch in range(start_epoch, self.num_epochs):
            self.logs.info(f"=== Epoch {epoch+1}/{self.num_epochs} ===")

            self.train_sampler.set_epoch(epoch)

            train_metrics = self._run_epoch(
                self.train_loader, training=True, epoch=epoch,
            )
            # Free fragmented CUDA memory before validation
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            val_metrics = self._run_epoch(
                self.val_loader, training=False, epoch=epoch,
            )
            
            # Update scheduler
            if epoch < scheduler_warmup_end:
                self.logs.info(
                    f"Scheduler warmup: skipping LR step (epoch {epoch+1}/{scheduler_warmup_end})"
                )
            else:
                self.scheduler.step()
            
            # Log peak VRAM usage after first epoch
            if epoch == start_epoch and self.device.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                self.logs.info(f"Peak VRAM usage: {peak_mb:.0f} MB")

            self._log_metrics("Train", train_metrics)
            self._log_metrics("Val  ", val_metrics)
            
            lr = self.optimizer.param_groups[0]['lr']
            self.logs.info(f"LR: {lr:.2e}")
            
            # TensorBoard logging
            self.tb_logger.log_epoch(epoch, train_metrics, val_metrics, lr)
            
            # Save best model (track recon loss only — total loss jumps when disc activates)
            val_recon = val_metrics.get('ReconSingle/Total') or val_metrics['loss']
            is_best = val_recon < best_val_loss
            if is_best:
                best_val_loss = val_recon
                self._save_checkpoint(epoch, val_metrics['loss'], best_val_loss, is_best=True)
                save_test_samples(
                    model=self.model,
                    dataset=self.val_dataset,
                    save_dir=self.save_path,
                    epoch=epoch,
                    sample_rate=self.sample_rate,
                    chunk_samples=self.chunk_samples,
                    num_samples=self.num_test_samples,
                    device=self.device,
                    logger=self.logs,
                    tb_logger=self.tb_logger if self.tb_log_audio else None,
                )
            
            # Periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch, val_metrics['loss'], best_val_loss)
            
            # Alpha sweep at designated epochs
            if epoch in alpha_sweep_epochs:
                self.tb_logger.run_and_log_alpha_sweep(
                    model=self.model,
                    dataset=self.val_dataset,
                    alphas=self.tb_alpha_sweep_alphas,
                    num_samples=self.tb_alpha_sweep_samples,
                    epoch=epoch,
                    device=self.device,
                    logger=self.logs,
                )

        self.logs.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
        self.tb_logger.close()
