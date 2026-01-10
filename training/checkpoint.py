"""Checkpoint saving and loading utilities."""

import os
from typing import Optional

import torch

from training.hub_utils import (
    resolve_checkpoint_path,
    upload_checkpoint_to_hub,
)


def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    model_config: dict,
    val_loss: float,
    best_val_loss: float,
    save_path: str,
    is_best: bool = False,
    logger: Optional[object] = None,
    hf_enabled: bool = False,
    hf_repo_id: Optional[str] = None,
    hf_push_best: bool = True,
    hf_push_checkpoints: bool = False,
    hf_push_interval: int = 5,
    hf_private: bool = False,
):
    """
    Save model checkpoint to disk and optionally upload to HuggingFace Hub.
    
    Args:
        epoch: Current epoch number (0-indexed)
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        scaler: AMP scaler state to save
        model_config: Model configuration dictionary
        val_loss: Current validation loss
        best_val_loss: Best validation loss so far
        save_path: Directory to save checkpoint
        is_best: Whether this is the best model so far
        logger: Optional logger for info messages
        hf_enabled: Whether HuggingFace Hub upload is enabled
        hf_repo_id: HuggingFace repository ID
        hf_push_best: Whether to push best model to Hub
        hf_push_checkpoints: Whether to push regular checkpoints to Hub
        hf_push_interval: Interval for pushing checkpoints to Hub
        hf_private: Whether Hub repository is private
    """
    ckpt = {
        'epoch': epoch,
        'model_config': model_config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
    }
    
    if is_best:
        path = os.path.join(save_path, "best_model.pth")
        torch.save(ckpt, path)
        if logger:
            logger.info(f"Best model saved: {path}")
        filename = "best_model.pth"
        should_upload = hf_enabled and hf_push_best
    else:
        path = os.path.join(save_path, f"checkpoint_{epoch+1}.pth")
        torch.save(ckpt, path)
        if logger:
            logger.info(f"Checkpoint saved: {path}")
        filename = f"checkpoint_{epoch+1}.pth"
        should_upload = (
            hf_enabled and 
            hf_push_checkpoints and 
            (epoch + 1) % hf_push_interval == 0
        )
    
    # Upload to HuggingFace Hub if enabled
    if should_upload and hf_repo_id:
        try:
            commit_message = (
                f"Upload {filename} (epoch {epoch+1}, val_loss={val_loss:.6f})"
            )
            success = upload_checkpoint_to_hub(
                checkpoint_path=path,
                repo_id=hf_repo_id,
                filename=filename,
                commit_message=commit_message,
                private=hf_private,
            )
            if success and logger:
                logger.info(f"Uploaded {filename} to HuggingFace Hub: {hf_repo_id}")
            elif not success and logger:
                logger.warning(f"Failed to upload {filename} to HuggingFace Hub")
        except Exception as e:
            if logger:
                logger.warning(f"Error uploading checkpoint to Hub: {e}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    reset_scheduler: bool = False,
    patience: int = 5,
    factor: float = 0.5,
    logger: Optional[object] = None,
) -> tuple:
    """
    Load checkpoint to resume training.
    
    Args:
        checkpoint_path: Path to checkpoint or HuggingFace Hub ID
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        scaler: AMP scaler to load state into
        device: Device to load checkpoint to
        reset_scheduler: Whether to reset scheduler state
        patience: Patience for new scheduler if resetting
        factor: Factor for new scheduler if resetting
        logger: Optional logger for info messages
    
    Returns:
        (start_epoch, best_val_loss): Tuple of starting epoch and best validation loss
    """
    if logger:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Resolve checkpoint path (download from Hub if necessary)
    try:
        resolved_path = resolve_checkpoint_path(checkpoint_path)
        if resolved_path != checkpoint_path and logger:
            logger.info(f"Downloaded checkpoint from Hub to: {resolved_path}")
    except FileNotFoundError as e:
        if logger:
            logger.error(f"Checkpoint not found: {e}")
        raise
    
    ckpt = torch.load(resolved_path, map_location=device, weights_only=False)
    
    # Use strict=False to allow loading older checkpoints missing new buffers
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing and logger:
        logger.info(f"Missing keys (using defaults): {missing}")
    if unexpected and logger:
        logger.info(f"Unexpected keys (ignored): {unexpected}")
    
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    # Handle scheduler state based on config
    if reset_scheduler:
        if logger:
            logger.info("Resetting scheduler state (loss landscape may have changed)")
        # Recreate scheduler with fresh state
        new_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            patience=patience, 
            factor=factor,
            threshold=1e-3,
            threshold_mode='rel',
            cooldown=1,
            min_lr=1e-6,
        )
        # Copy new scheduler state to existing scheduler
        scheduler.load_state_dict(new_scheduler.state_dict())
    else:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    
    if 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    
    if logger:
        logger.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
    
    return start_epoch, best_val_loss

