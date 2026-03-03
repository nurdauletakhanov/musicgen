"""Core Trainer class for STFT Autoencoder training."""

import os

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from data.dataloader import build_dataloaders
from data.dataloader_musdb import build_musdb_dataloaders
from evaluation.samples import save_test_samples
from models.autoencoder import Autoencoder
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
        self.grad_accum_steps = train_cfg.get('gradient_accumulation_steps', 1)
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

        # Data loaders
        self.use_musdb_single = data_cfg.get('use_musdb_single', False)

        if not self.use_musdb_single:
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
        else:
            from data.dataloader_musdb import build_musdb_single_stem_dataloaders
            (
                self.train_loader,
                self.val_loader,
                self.train_dataset,
                self.val_dataset,
                self.train_sampler,
            ) = build_musdb_single_stem_dataloaders(
                chunks_dir=data_cfg['musdb_chunks_dir'],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                device=self.device,
                logger=self.logs,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
            )
            self.logs.info("Primary data: MUSDB18 single stems")

        self.logs.info(
            f"Data loading: num_workers={self.num_workers}, "
            f"prefetch_factor={self.prefetch_factor}, "
            f"pin_memory={self.pin_memory}, "
            f"persistent_workers={self.persistent_workers}"
        )

        # MUSDB18 stem pair loaders (optional)
        self.use_stem_pairs = data_cfg.get('use_stem_pairs', False)
        musdb_chunks_dir = data_cfg.get('musdb_chunks_dir', None)
        self.musdb_train_loader = None
        self.musdb_val_loader = None
        self.musdb_train_sampler = None

        if self.use_stem_pairs and musdb_chunks_dir:
            (
                self.musdb_train_loader,
                self.musdb_val_loader,
                self.musdb_train_dataset,
                self.musdb_val_dataset,
                self.musdb_train_sampler,
            ) = build_musdb_dataloaders(
                chunks_dir=musdb_chunks_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                device=self.device,
                logger=self.logs,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
            )
            self.logs.info("MUSDB18 stem pair mixing: enabled")
        
        # AMP setup — use bf16 (wider dynamic range, no GradScaler needed)
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16
        self.scaler = GradScaler("cuda", enabled=False)  # disabled for bf16
        self.logs.info(f"Mixed precision (AMP): {'bf16' if self.use_amp else 'disabled'}")
        
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
        """Build model, optimizer, and scheduler."""
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

    def load_checkpoint(self, checkpoint_path: str) -> tuple:
        """Load checkpoint to resume training."""
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
            reset_scheduler=self.reset_scheduler_on_resume,
            logger=self.logs,
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

    def _run_epoch(self, loader, training: bool = True, musdb_loader=None) -> dict:
        """
        Run one epoch of training or validation.

        Args:
            loader: Primary data loader (MAESTRO)
            training: Whether this is a training epoch
            musdb_loader: Optional MUSDB18 stem pair loader for mixing loss

        Returns dict with:
        - Mean values for all metrics
        - Raw MixRate values for distribution stats (validation only)
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        # Metric collectors
        metric_keys = [
            'loss',
            # ReconSingle
            'ReconSingle/Total', 'ReconSingle/WavL1', 'ReconSingle/MRSTFT',
            # MixReconInterp
            'MixReconInterp/Total', 'MixReconInterp/WavL1', 'MixReconInterp/MRSTFT',
            # MixReconReal
            'MixReconReal/Total', 'MixReconReal/WavL1', 'MixReconReal/MRSTFT',
            # Key metrics
            'MixRate', 'MixGap', 'LatentMixError',
            # Latent space stats
            'Latent/mean', 'Latent/std', 'Latent/absmax',
            # Gradient norm (train only)
            'grad_norm',
            # Stem mixing metrics
            'StemMix/Total', 'StemMix/Rate', 'StemMix/Gap',
            # Legacy
            'latent_mix',
        ]
        metrics = {k: [] for k in metric_keys}

        # Collect raw MixRate values for distribution stats
        mix_rate_values = []

        desc = "Train" if training else "Val"
        context = torch.enable_grad() if training else torch.no_grad()

        # Set up musdb iterator if available
        musdb_iter = iter(musdb_loader) if musdb_loader is not None else None

        with context:
            for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
                x_stft = batch['x_stft'].to(self.device, non_blocking=True)
                x_wave = batch['x_wave'].to(self.device, non_blocking=True)

                if x_wave.dim() == 2:
                    x_wave = x_wave.unsqueeze(1)

                with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    total, components = self.model(x_stft, x_wave)

                    # Stem pair mixing loss from MUSDB18
                    if musdb_iter is not None and self.model.decode_mix_weight > 0:
                        try:
                            musdb_batch = next(musdb_iter)
                        except StopIteration:
                            musdb_iter = iter(musdb_loader)
                            musdb_batch = next(musdb_iter)

                        m_stft1 = musdb_batch['x_stft'].to(self.device, non_blocking=True)
                        m_wave1 = musdb_batch['x_wave'].to(self.device, non_blocking=True)
                        m_stft2 = musdb_batch['x_stft2'].to(self.device, non_blocking=True)
                        m_wave2 = musdb_batch['x_wave2'].to(self.device, non_blocking=True)

                        stem_mix_dict = self.model.compute_stem_mixing_loss(
                            m_stft1, m_wave1, m_stft2, m_wave2
                        )
                        stem_mix_loss = stem_mix_dict['total']
                        total = total + self.model.decode_mix_weight * stem_mix_loss

                        components['StemMix/Total'] = stem_mix_loss.detach().item()
                        components['StemMix/Rate'] = stem_mix_dict.get('rate', 0.0)
                        components['StemMix/Gap'] = stem_mix_dict.get('gap', 0.0)

                if training:
                    scaled_loss = total / self.grad_accum_steps
                    scaled_loss.backward()

                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        metrics['grad_norm'].append(grad_norm.item())
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                # Collect metrics
                metrics['loss'].append(total.item())
                for key in metrics.keys():
                    if key != 'loss' and key in components:
                        metrics[key].append(components[key])

                # Collect raw MixRate for distribution stats
                mix_rate = components.get('MixRate', 0.0)
                if mix_rate > 0:  # Only collect non-zero rates (mixing enabled)
                    mix_rate_values.append(mix_rate)

        # Flush any remaining accumulated gradients at end of epoch
        if training and self.grad_accum_steps > 1:
            num_batches = batch_idx + 1
            if num_batches % self.grad_accum_steps != 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                metrics['grad_norm'].append(grad_norm.item())
                self.optimizer.step()
                self.optimizer.zero_grad()

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
            scaler=self.scaler,
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
        )

    def _log_metrics(self, prefix: str, metrics: dict):
        """Log training/validation metrics to console."""
        recon_total = metrics.get('ReconSingle/Total', 0.0)
        recon_l1 = metrics.get('ReconSingle/WavL1', 0.0)
        recon_mrstft = metrics.get('ReconSingle/MRSTFT', 0.0)
        mix_interp = metrics.get('MixReconInterp/Total', 0.0)
        mix_real = metrics.get('MixReconReal/Total', 0.0)
        mix_rate = metrics.get('MixRate', 0.0)
        mix_gap = metrics.get('MixGap', 0.0)
        latent_err = metrics.get('LatentMixError', 0.0)
        
        self.logs.info(
            f"{prefix} - Loss: {metrics['loss']:.6f} | "
            f"Recon: {recon_total:.4f} (L1: {recon_l1:.4f}, MR: {recon_mrstft:.4f}) | "
            f"MixInterp: {mix_interp:.4f}, MixReal: {mix_real:.4f} | "
            f"Rate: {mix_rate:.4f}, Gap: {mix_gap:.4f} | "
            f"LatErr: {latent_err:.6f}"
        )

    def fit(self, start_epoch: int = 0, best_val_loss: float = float('inf')):
        """Main training loop."""
        scheduler_warmup_end = start_epoch + self.scheduler_warmup_epochs if start_epoch > 0 else 0
        
        # Log hyperparameters once at start
        self.tb_logger.log_training_config(self.cfg, self.model_config)
        
        # Calculate alpha sweep epochs (1/3, 2/3, final)
        alpha_sweep_epochs = get_alpha_sweep_epochs(self.num_epochs)
        
        for epoch in range(start_epoch, self.num_epochs):
            self.logs.info(f"=== Epoch {epoch+1}/{self.num_epochs} ===")
            
            self.train_sampler.set_epoch(epoch)
            if self.musdb_train_sampler is not None:
                self.musdb_train_sampler.set_epoch(epoch)

            train_metrics = self._run_epoch(
                self.train_loader, training=True,
                musdb_loader=self.musdb_train_loader,
            )
            val_metrics = self._run_epoch(
                self.val_loader, training=False,
                musdb_loader=self.musdb_val_loader,
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
            
            # Save best model
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
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
                    use_amp=self.use_amp,
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
                    use_amp=self.use_amp,
                    logger=self.logs,
                )

        self.logs.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
        self.tb_logger.close()
