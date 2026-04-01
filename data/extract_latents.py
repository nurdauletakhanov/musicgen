"""Extract latent tokens from audio chunks using a frozen autoencoder.

Usage:
    python -m data.extract_latents \
        --checkpoint checkpoints/comp-24x-v6-phase2-mix0.01/best_model.pth \
        --chunks_dir ./musdb-chunks-stft-5s \
        --output_dir ./latent-cache/musdb-5s \
        --splits train val test

Can be run multiple times with different --chunks_dir to accumulate
latents from different datasets (MUSDB18, FMA, Jamendo, etc.).
"""

import argparse
import os

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataloader import SingleStemDataset
from models.autoencoder import Autoencoder


def extract(checkpoint_path, chunks_dir, output_dir, splits, batch_size=32,
            gpu_index=0):
    """Encode all chunks through frozen encoder and save latents."""
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available()
                          else 'cpu')
    use_amp = device.type == "cuda"

    # Load autoencoder
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_cfg = ckpt['model_config']
    ae = Autoencoder(**model_cfg)
    ae.load_state_dict(ckpt['model_state_dict'])
    ae = ae.to(device)
    ae.eval()

    print(f"Loaded autoencoder from {checkpoint_path}")
    print(f"Latent shape: [{model_cfg.get('num_segments', '?')}, "
          f"{model_cfg.get('d_model', '?')}]")

    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        split_dir = os.path.join(chunks_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split} — not found: {split_dir}")
            continue

        save_file = os.path.join(output_dir, f'latents_{split}.pt')
        if os.path.exists(save_file):
            existing = torch.load(save_file, weights_only=True)
            print(f"Already exists: {save_file} "
                  f"({existing['latents'].shape[0]} latents)")
            continue

        dataset = SingleStemDataset(chunks_dir, split=split)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        all_latents = []
        all_stems = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Encoding {split}"):
                x_stft = batch['x_stft'].to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    z = ae.encoder(x_stft)
                all_latents.append(z.cpu())

                if 'stem' in batch:
                    all_stems.extend(batch['stem'])

        latents = torch.cat(all_latents, dim=0)
        print(f"{split}: {latents.shape[0]} latents, shape {list(latents.shape[1:])}")

        # Save latent stats for normalization (mean/std across dataset)
        mean = latents.mean(dim=0)
        std = latents.std(dim=0)

        torch.save({
            'latents': latents,
            'stems': all_stems,
            'mean': mean,
            'std': std,
            'ae_checkpoint': checkpoint_path,
            'chunks_dir': chunks_dir,
        }, save_file)
        print(f"Saved: {save_file}")

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract latents from audio chunks")
    parser.add_argument('--checkpoint', required=True,
                        help='Path to autoencoder checkpoint')
    parser.add_argument('--chunks_dir', required=True,
                        help='Path to preprocessed audio chunks')
    parser.add_argument('--output_dir', required=True,
                        help='Where to save latent .pt files')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Which splits to extract')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    extract(args.checkpoint, args.chunks_dir, args.output_dir, args.splits,
            args.batch_size, args.gpu_index)
