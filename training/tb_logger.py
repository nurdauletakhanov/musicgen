"""TensorBoard logging utilities for linearity research."""

import os
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class TBLogger:
    """
    TensorBoard logger with paper-ready metric naming.
    
    Supports:
    - Scalar metrics with train/val prefixes
    - Distribution statistics (mean, median, p90, max)
    - Audio samples
    - Hyperparameter logging
    - Fixed alpha sweep results
    """
    
    def __init__(self, log_dir: str, enabled: bool = True):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard events
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.log_dir = log_dir
        self.writer = None
        
        if self.enabled:
            tb_dir = os.path.join(log_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
    
    def log_scalar(self, name: str, value: float, step: int, prefix: str = ""):
        """Log a single scalar value."""
        if not self.enabled or self.writer is None:
            return
        
        tag = f"{prefix}/{name}" if prefix else name
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log multiple scalar metrics.
        
        Args:
            metrics: Dict of metric_name -> value
            step: Training step (epoch)
            prefix: Prefix for all metrics (e.g., "train" or "val")
        """
        if not self.enabled or self.writer is None:
            return
        
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.writer.add_scalar(tag, value, step)
    
    def log_distribution(
        self, 
        name: str, 
        values: List[float], 
        step: int, 
        prefix: str = ""
    ):
        """
        Log distribution statistics: mean, median, p90, max.
        
        Args:
            name: Base metric name (e.g., "MixRate")
            values: List of per-sample values
            step: Training step (epoch)
            prefix: Prefix for metrics (e.g., "val")
        """
        if not self.enabled or self.writer is None:
            return
        
        if len(values) == 0:
            return
        
        arr = np.array(values)
        stats = {
            f"{name}/mean": float(np.mean(arr)),
            f"{name}/median": float(np.median(arr)),
            f"{name}/p90": float(np.percentile(arr, 90)),
            f"{name}/max": float(np.max(arr)),
        }
        
        self.log_scalars(stats, step, prefix)
    
    def log_audio(
        self, 
        name: str, 
        audio: torch.Tensor, 
        step: int, 
        sample_rate: int,
        prefix: str = ""
    ):
        """
        Log audio sample to TensorBoard.
        
        Args:
            name: Audio sample name
            audio: Audio tensor [L] or [1, L]
            step: Training step (epoch)
            sample_rate: Audio sample rate
            prefix: Prefix for tag
        """
        if not self.enabled or self.writer is None:
            return
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Ensure float and clamp to valid range
        audio = audio.float().clamp(-1.0, 1.0)
        
        tag = f"{prefix}/{name}" if prefix else name
        self.writer.add_audio(tag, audio, step, sample_rate=sample_rate)
    
    def log_hparams(self, hparams: Dict, metrics: Optional[Dict] = None):
        """
        Log hyperparameters and optional final metrics.
        
        Args:
            hparams: Dict of hyperparameter name -> value
            metrics: Optional dict of final metrics for hparam comparison
        """
        if not self.enabled or self.writer is None:
            return
        
        # Flatten nested dicts and convert to strings if needed
        flat_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_hparams[f"{key}/{sub_key}"] = self._to_hparam_value(sub_value)
            else:
                flat_hparams[key] = self._to_hparam_value(value)
        
        if metrics is None:
            metrics = {}
        
        self.writer.add_hparams(flat_hparams, metrics)
    
    def log_alpha_sweep(
        self, 
        alpha: float, 
        metrics: Dict[str, float], 
        step: int
    ):
        """
        Log fixed alpha sweep results.
        
        Args:
            alpha: Fixed alpha value (e.g., 0.1, 0.3, 0.5)
            metrics: Dict with MixReconInterp, MixRate, etc.
            step: Training step (epoch)
        """
        if not self.enabled or self.writer is None:
            return
        
        # Format alpha for tag (e.g., "alpha_0.1")
        alpha_tag = f"alpha_{alpha:.1f}"
        
        for name, value in metrics.items():
            tag = f"AlphaSweep/{alpha_tag}/{name}"
            self.writer.add_scalar(tag, value, step)
    
    def _to_hparam_value(self, value):
        """Convert value to hparam-compatible type."""
        if isinstance(value, (list, tuple)):
            return str(value)
        if isinstance(value, (int, float, str, bool)):
            return value
        return str(value)
    
    def flush(self):
        """Flush pending events to disk."""
        if self.writer is not None:
            self.writer.flush()
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
    
    # -------------------------
    # High-level logging methods
    # -------------------------
    
    def log_epoch(
        self, 
        epoch: int, 
        train_metrics: Dict, 
        val_metrics: Dict,
        learning_rate: float
    ):
        """
        Log all epoch metrics to TensorBoard.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dict (may contain _mix_rate_values)
            val_metrics: Validation metrics dict (may contain _mix_rate_values)
            learning_rate: Current learning rate
        """
        if not self.enabled:
            return
        
        # Filter out internal keys
        def filter_metrics(m):
            return {k: v for k, v in m.items() if not k.startswith('_')}
        
        self.log_scalars(filter_metrics(train_metrics), epoch, prefix="train")
        self.log_scalars(filter_metrics(val_metrics), epoch, prefix="val")
        self.log_scalar("lr", learning_rate, epoch)
        
        # Log MixRate distribution stats (validation only)
        mix_rate_values = val_metrics.get('_mix_rate_values', [])
        if mix_rate_values:
            self.log_distribution("MixRate", mix_rate_values, epoch, prefix="val")
        
        self.flush()
    
    def log_training_config(self, cfg: Dict, model_config: Dict):
        """
        Log hyperparameters from config at training start.
        
        Args:
            cfg: Full training config dict
            model_config: Model configuration dict
        """
        if not self.enabled:
            return
        
        data_cfg = cfg.get('data', {})
        train_cfg = cfg.get('train', {})
        
        hparams = {
            'dataset': data_cfg.get('chunks_dir', 'unknown'),
            'sample_rate': data_cfg.get('sample_rate', 44100),
            'chunk_seconds': data_cfg.get('chunk_seconds', 1.0),
            'batch_size': train_cfg.get('batch_size', 32),
            'learning_rate': train_cfg.get('learning_rate', 1e-4),
            'num_epochs': train_cfg.get('num_epochs', 40),
            **{f'model/{k}': v for k, v in model_config.items()},
        }
        self.log_hparams(hparams)
    
    def run_and_log_alpha_sweep(
        self,
        model: torch.nn.Module,
        dataset: "Dataset",
        alphas: List[float],
        num_samples: int,
        epoch: int,
        device: torch.device,
        logger=None,
    ):
        """
        Run fixed alpha sweep and log results to TensorBoard.

        Args:
            model: Autoencoder model
            dataset: Validation dataset
            alphas: List of fixed alpha values
            num_samples: Number of sample pairs to evaluate
            epoch: Current epoch number
            device: Device for inference
            logger: Optional text logger for console output
        """
        if not self.enabled:
            return

        from evaluation.alpha_sweep import run_alpha_sweep

        if logger:
            logger.info(f"Running alpha sweep evaluation at epoch {epoch + 1}...")

        results = run_alpha_sweep(
            model=model,
            dataset=dataset,
            alphas=alphas,
            num_samples=num_samples,
            device=device,
        )
        
        for alpha, metrics in results.items():
            self.log_alpha_sweep(alpha, metrics, epoch)
            if logger:
                logger.info(
                    f"  alpha={alpha:.1f}: MixReconInterp={metrics['MixReconInterp']:.4f}, "
                    f"MixRate={metrics['MixRate']:.4f}"
                )
        
        self.flush()


def get_alpha_sweep_epochs(num_epochs: int) -> set:
    """
    Get epochs at which to run alpha sweep (1/3, 2/3, final).
    
    Args:
        num_epochs: Total number of training epochs
        
    Returns:
        Set of epoch indices (0-based) for alpha sweep
    """
    epochs = set()
    if num_epochs >= 3:
        epochs.add(num_epochs // 3 - 1)
        epochs.add(2 * num_epochs // 3 - 1)
    epochs.add(num_epochs - 1)  # Final epoch
    return epochs

