"""
Entry point for STFT Autoencoder training.

Usage:
    python -m training.train --config configs/experiments/current.yaml
    python -m training.train --config configs/experiments/current.yaml --resume checkpoints/my-exp/checkpoint_10.pth
"""

import argparse

from training.config import copy_config_to_checkpoint
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train STFT Autoencoder")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/experiments/current.yaml",
        help="Path to config YAML (default: configs/experiments/current.yaml)"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="Path to base config YAML (default: auto-detect configs/base.yaml)"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None, 
        help="Checkpoint to resume from (local path or HuggingFace Hub ID)"
    )
    args = parser.parse_args()

    trainer = Trainer(args.config, base_config_path=args.base_config)
    trainer.build_model()
    
    # Copy config to checkpoint directory for reproducibility (only for new runs)
    if not args.resume:
        copy_config_to_checkpoint(args.config, trainer.save_path)
    
    if args.resume:
        start_epoch, best_val_loss = trainer.load_checkpoint(args.resume)
        if trainer.cfg.get('train', {}).get('reset_best_loss', False):
            trainer.logs.info(
                f"reset_best_loss=True: resetting best_val_loss from "
                f"{best_val_loss:.6f} to inf"
            )
            best_val_loss = float('inf')
        trainer.fit(start_epoch=start_epoch, best_val_loss=best_val_loss)
    else:
        trainer.fit()


if __name__ == "__main__":
    main()
