"""Entry point for latent diffusion training.

Usage:
    # Step 1: Extract latents (one-time)
    python -m data.extract_latents \
        --checkpoint checkpoints/comp-24x-v6-phase2-mix0.01/best_model.pth \
        --chunks_dir ./musdb-chunks-stft-5s \
        --output_dir ./latent-cache/musdb-5s

    # Step 2: Train diffusion
    python train_diffusion.py --config configs/experiments/diffusion/dit_v1.yaml
"""

import argparse

from training.diffusion_trainer import DiffusionTrainer


def main():
    parser = argparse.ArgumentParser(description="Train latent diffusion model")
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--resume', default=None, help='Checkpoint to resume from')
    args = parser.parse_args()

    trainer = DiffusionTrainer(args.config)
    trainer.build_model()
    trainer.prepare_data()

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        ckpt = __import__('torch').load(args.resume, map_location='cpu',
                                         weights_only=False)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            trainer.lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    trainer.fit(start_epoch, best_val_loss)


if __name__ == '__main__':
    main()
