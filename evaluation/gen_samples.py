"""
Quick audio quality check: encode + decode test samples and save to disk.

Usage:
    python -m evaluation.gen_samples
    python -m evaluation.gen_samples --checkpoint checkpoints/comp-46x-v2/best_model.pth
    python -m evaluation.gen_samples --num-samples 5 --output results/samples/
"""

import argparse
import os

import soundfile as sf
import torch
from torch.amp import autocast

from evaluation.utils import load_model, load_test_singles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/comp-46x-v2/best_model.pth")
    parser.add_argument("--chunks-dir", default="./musdb-chunks-stft-5s")
    parser.add_argument("--output", default="results/samples/comp-46x-v2")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  Epoch: {ckpt.get('epoch', '?')}  Val loss: {ckpt.get('val_loss', '?'):.4f}")

    print(f"Loading test samples from {args.chunks_dir}...")
    samples = load_test_singles(args.chunks_dir, num_samples=args.num_samples)
    print(f"  Loaded {len(samples)} samples")

    os.makedirs(args.output, exist_ok=True)

    tgt = model.decoder.target_length

    with torch.no_grad():
        for i, sample in enumerate(samples):
            x_stft = sample["x_stft"].unsqueeze(0).to(device)
            x_wave = sample["x_wave"].unsqueeze(0).to(device)
            if x_wave.dim() == 2:
                x_wave = x_wave.unsqueeze(1)

            with autocast("cuda", enabled=True):
                z = model.encoder(x_stft)
                y_hat, _ = model.decoder(z)

            orig = x_wave[0, 0, :tgt].cpu().float()
            recon = y_hat[0, 0].cpu().float()

            # Peak-normalize both
            for t in (orig, recon):
                peak = t.abs().max()
                if peak > 0.95:
                    t.div_(peak / 0.95)

            tag = f"{sample['track'][:20]}_{sample['stem']}".replace(" ", "_").replace("/", "_")
            sf.write(os.path.join(args.output, f"{i:02d}_{tag}_original.wav"),
                     orig.numpy(), args.sample_rate, subtype="PCM_16")
            sf.write(os.path.join(args.output, f"{i:02d}_{tag}_reconstructed.wav"),
                     recon.numpy(), args.sample_rate, subtype="PCM_16")
            print(f"  [{i+1}/{len(samples)}] {tag}")

    print(f"\nDone. Files saved to {args.output}/")


if __name__ == "__main__":
    main()
