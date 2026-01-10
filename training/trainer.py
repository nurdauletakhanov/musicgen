"""Core Trainer class for STFT Autoencoder training."""

import os

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from data.dataloader import build_dataloaders
from evaluation.samples import save_test_samples
from models.autoencoder import Autoencoder
from training.checkpoint import save_checkpoint, load_checkpoint
from training.config import get_device, load_config, build_model_config
from training.hub_utils import (
    check_authentication,
    get_repo_id_from_config,
    upload_config_to_hub,
)
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
        self.patience = train_cfg.get('patience', 5)
        self.factor = train_cfg.get('factor', 0.5)
        self.save_interval = train_cfg.get('save_interval', 10)
        self.num_workers = train_cfg.get('num_workers', 4)  
        self.pin_memory = train_cfg.get('pin_memory', None)
        self.persistent_workers = train_cfg.get('persistent_workers', None)
        self.prefetch_factor = train_cfg.get('prefetch_factor', 2)
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

        # Audio params for sample saving
        self.sample_rate = data_cfg.get('sample_rate', 44100)
        chunk_seconds = data_cfg.get('chunk_seconds', 1.0)
        self.chunk_samples = int(self.sample_rate * chunk_seconds)
        self.num_test_samples = train_cfg.get('num_test_samples', 5)
        
        # Build model config
        self.model_config = build_model_config(cfg)

        # Data loaders
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
        
        self.logs.info(
            f"Data loading: num_workers={self.num_workers}, "
            f"prefetch_factor={self.prefetch_factor}, "
            f"pin_memory={self.pin_memory}, "
            f"persistent_workers={self.persistent_workers}"
        )
        
        # AMP setup
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.logs.info(f"Mixed precision (AMP): {'enabled' if self.use_amp else 'disabled'}")
        
        # HuggingFace Hub setup
        self._setup_hub(hf_cfg)

    def _setup_hub(self, hf_cfg: dict):
        """Initialize HuggingFace Hub integration."""
        self.hf_enabled = hf_cfg.get('enabled', False)
        self.hf_repo_id = hf_cfg.get('repo_id', None)
        self.hf_push_best = hf_cfg.get('push_best', True)
        self.hf_push_checkpoints = hf_cfg.get('push_checkpoints', False)
        self.hf_push_interval = hf_cfg.get('push_interval', 5)
        self.hf_private = hf_cfg.get('private', False)
        
        if self.hf_enabled:
            is_auth, username = check_authentication()
            if not is_auth:
                self.logs.warning(
                    "HuggingFace Hub integration enabled but not authenticated. "
                    "Run: huggingface-cli login"
                )
                self.hf_enabled = False
            else:
                if self.hf_repo_id is None:
                    try:
                        config_name = self.cfg.get('name', 'default')
                        self.hf_repo_id = get_repo_id_from_config(config_name, username)
                    except Exception as e:
                        self.logs.warning(
                            f"Could not auto-generate repo_id: {e}. Disabling Hub integration."
                        )
                        self.hf_enabled = False
                
                if self.hf_enabled:
                    self.logs.info(f"HuggingFace Hub integration enabled. Repo ID: {self.hf_repo_id}")
                    try:
                        upload_config_to_hub(
                            self.config_path,
                            self.hf_repo_id,
                            commit_message="Initial config upload"
                        )
                    except Exception as e:
                        self.logs.warning(f"Could not upload config to Hub: {e}")

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
            patience=self.patience,
            factor=self.factor,
            logger=self.logs,
        )
        if self.scheduler_warmup_epochs > 0:
            self.logs.info(
                f"Scheduler warmup enabled: LR reduction disabled for "
                f"{self.scheduler_warmup_epochs} epochs after resume"
            )
        return start_epoch, best_val_loss

    def _run_epoch(self, loader, training: bool = True) -> dict:
        """Run one epoch of training or validation."""
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        metrics = {k: [] for k in [
            'loss', 'recon', 'wav_l1', 'mrstft', 'latent_mix',
            'decode_mix', 'decode_mix_l1', 'decode_mix_mrstft', 'decode_mix_rate'
        ]}
        
        desc = "Train" if training else "Val"
        context = torch.enable_grad() if training else torch.no_grad()
        
        with context:
            for batch in tqdm(loader, desc=desc, leave=False):
                x_stft = batch['x_stft'].to(self.device, non_blocking=True)
                x_wave = batch['x_wave'].to(self.device, non_blocking=True)
                
                if x_wave.dim() == 2:
                    x_wave = x_wave.unsqueeze(1)
                
                with autocast("cuda", enabled=self.use_amp):
                    total, components = self.model(x_stft, x_wave)

                if training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(total).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                metrics['loss'].append(total.item())
                metrics['recon'].append(components.get('recon', 0.0))
                metrics['wav_l1'].append(components.get('wav_l1', 0.0))
                metrics['mrstft'].append(components.get('mrstft', 0.0))
                metrics['latent_mix'].append(components.get('latent_mix', 0.0))
                metrics['decode_mix'].append(components.get('decode_mix', 0.0))
                metrics['decode_mix_l1'].append(components.get('decode_mix_l1', 0.0))
                metrics['decode_mix_mrstft'].append(components.get('decode_mix_mrstft', 0.0))
                metrics['decode_mix_rate'].append(components.get('rate', 0.0))

        return {k: np.mean(v) for k, v in metrics.items()}

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
        """Log training/validation metrics."""
        self.logs.info(
            f"{prefix} - Loss: {metrics['loss']:.6f}, "
            f"Recon: {metrics['recon']:.6f}, "
            f"WavL1: {metrics['wav_l1']:.6f}, "
            f"MRSTFT: {metrics['mrstft']:.6f}, "
            f"LatMix: {metrics['latent_mix']:.6f}, "
            f"DecMix: {metrics['decode_mix']:.6f}, "
            f"DecMixL1: {metrics['decode_mix_l1']:.6f}, "
            f"DecMixMRSTFT: {metrics['decode_mix_mrstft']:.6f}, "
            f"DecMixRate: {metrics['decode_mix_rate']:.4f}"
        )

    def fit(self, start_epoch: int = 0, best_val_loss: float = float('inf')):
        """Main training loop."""
        scheduler_warmup_end = start_epoch + self.scheduler_warmup_epochs if start_epoch > 0 else 0
        
        for epoch in range(start_epoch, self.num_epochs):
            self.logs.info(f"=== Epoch {epoch+1}/{self.num_epochs} ===")
            
            self.train_sampler.set_epoch(epoch)
            
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)
            
            # Update scheduler
            if epoch < scheduler_warmup_end:
                self.logs.info(
                    f"Scheduler warmup: skipping LR step (epoch {epoch+1}/{scheduler_warmup_end})"
                )
            else:
                self.scheduler.step(val_metrics['loss'])
            
            self._log_metrics("Train", train_metrics)
            self._log_metrics("Val  ", val_metrics)
            self.logs.info(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
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
                )
            
            # Periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch, val_metrics['loss'], best_val_loss)

        self.logs.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
