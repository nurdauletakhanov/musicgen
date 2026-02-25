"""
Generate mixing demonstration audio samples for NeurIPS paper supplementary.

For each demo, generates 5 audio files:
1. stem_A.wav - First stem (e.g., drums)
2. stem_B.wav - Second stem (e.g., vocals)
3. ground_truth_mix.wav - Actual A + B waveform
4. oracle_mix.wav - D(E(A + B)) - encode-decode the actual mix
5. latent_mix.wav - D(E(A) + E(B)) - decode the sum of latents

The key comparison is between oracle_mix and latent_mix:
- If they sound the same, the latent space is linear!

Usage:
    python -m evaluation.generate_mixing_demos \
        --checkpoint checkpoints/musdb-phase2-mixing/best_model.pth \
        --output results/demos/
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.amp import autocast

from models.autoencoder import Autoencoder


def load_model(checkpoint_path: str, device: torch.device) -> Autoencoder:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Autoencoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def load_stem_data(chunks_dir: str, split: str = "test") -> Dict[str, Dict]:
    """
    Load all stem data from MUSDB chunks.

    Returns dict mapping track_key -> {stem_name -> {x_stft, x_wave}}.
    """
    index_path = os.path.join(chunks_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    split_index = index.get(split, {})
    if not split_index:
        raise RuntimeError(f"No data for split '{split}'")

    stem_order = ["drums", "bass", "other", "vocals"]
    tracks = {}

    for track_key, track_info in split_index.items():
        if not isinstance(track_info, dict) or "stems" not in track_info:
            continue

        stems = track_info["stems"]
        n_chunks = track_info.get("num_chunks", 0)

        # Check all stems exist
        if not all(s in stems for s in stem_order):
            continue

        # Load stem data
        stem_data = {}
        for stem_name in stem_order:
            path = os.path.join(chunks_dir, split, stems[stem_name])
            if not os.path.exists(path):
                break
            data = torch.load(path, map_location="cpu", weights_only=True)
            stem_data[stem_name] = {
                "x_stft": data["x_stft"].float(),
                "x_wave": data["x_wave"].float(),
            }

        if len(stem_data) == 4:
            tracks[track_key] = {
                "stems": stem_data,
                "num_chunks": n_chunks,
            }

    return tracks


def generate_demo(
    model: torch.nn.Module,
    stem1_data: Dict,
    stem2_data: Dict,
    stem1_name: str,
    stem2_name: str,
    chunk_idx: int,
    output_dir: str,
    demo_name: str,
    sample_rate: int,
    device: torch.device,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Generate a single mixing demo.

    Creates 5 audio files and returns metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load chunk data
    x1_stft = stem1_data["x_stft"][chunk_idx].unsqueeze(0).to(device)
    x1_wave = stem1_data["x_wave"][chunk_idx].unsqueeze(0).to(device)
    x2_stft = stem2_data["x_stft"][chunk_idx].unsqueeze(0).to(device)
    x2_wave = stem2_data["x_wave"][chunk_idx].unsqueeze(0).to(device)

    if x1_wave.dim() == 2:
        x1_wave = x1_wave.unsqueeze(1)
        x2_wave = x2_wave.unsqueeze(1)

    beta = 1.0 - alpha

    with torch.no_grad():
        with autocast("cuda", enabled=True):
            # Encode individual stems
            z1 = model.encoder(x1_stft)
            z2 = model.encoder(x2_stft)

            # Ground truth mix (waveform domain)
            x_mix_wave = alpha * x1_wave + beta * x2_wave

            # Oracle: D(E(A + B))
            x_mix_stft = model._compute_stft_from_wave(x_mix_wave)
            z_mix = model.encoder(x_mix_stft)
            x_oracle = model.decoder(z_mix)

            # Latent mix: D(alpha * E(A) + beta * E(B))
            z_latent_mix = alpha * z1 + beta * z2
            x_latent = model.decoder(z_latent_mix)

            # Compute metrics
            l1_oracle = F.l1_loss(x_oracle, x_mix_wave).item()
            l1_latent = F.l1_loss(x_latent, x_mix_wave).item()
            mr_oracle = model.mrstft_loss(x_oracle, x_mix_wave).item()
            mr_latent = model.mrstft_loss(x_latent, x_mix_wave).item()

    # Convert to numpy for saving
    def to_audio(tensor):
        audio = tensor[0, 0].cpu().float().numpy()
        # Normalize to prevent clipping
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 0.99:
            audio = audio / max_val * 0.95
        return audio

    x1_audio = to_audio(x1_wave)
    x2_audio = to_audio(x2_wave)
    mix_audio = to_audio(x_mix_wave)
    oracle_audio = to_audio(x_oracle)
    latent_audio = to_audio(x_latent)

    # Save audio files
    sf.write(os.path.join(output_dir, f"{demo_name}_stem_A_{stem1_name}.wav"),
             x1_audio, sample_rate, subtype="PCM_16")
    sf.write(os.path.join(output_dir, f"{demo_name}_stem_B_{stem2_name}.wav"),
             x2_audio, sample_rate, subtype="PCM_16")
    sf.write(os.path.join(output_dir, f"{demo_name}_ground_truth_mix.wav"),
             mix_audio, sample_rate, subtype="PCM_16")
    sf.write(os.path.join(output_dir, f"{demo_name}_oracle_mix.wav"),
             oracle_audio, sample_rate, subtype="PCM_16")
    sf.write(os.path.join(output_dir, f"{demo_name}_latent_mix.wav"),
             latent_audio, sample_rate, subtype="PCM_16")

    return {
        "l1_oracle": l1_oracle,
        "l1_latent": l1_latent,
        "mr_oracle": mr_oracle,
        "mr_latent": mr_latent,
        "mix_rate": (l1_latent + mr_latent) / (l1_oracle + mr_oracle + 1e-8),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate mixing demo audio samples for NeurIPS supplementary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="./musdb-chunks-stft",
        help="Directory containing preprocessed MUSDB chunks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/demos",
        help="Output directory for demo audio files"
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=10,
        help="Number of demos to generate"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Audio sample rate"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mixing ratio (0.5 = equal blend)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to use"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)

    # Load stem data
    print(f"Loading stem data from {args.chunks_dir}...")
    tracks = load_stem_data(args.chunks_dir, args.split)
    print(f"Found {len(tracks)} tracks with all 4 stems")

    if not tracks:
        print("Error: No valid tracks found!")
        return

    # Define stem pairs to demo
    stem_pairs = [
        ("drums", "vocals"),
        ("drums", "bass"),
        ("bass", "vocals"),
        ("other", "vocals"),
        ("drums", "other"),
        ("bass", "other"),
    ]

    # Generate demos
    os.makedirs(args.output, exist_ok=True)

    demo_count = 0
    all_metrics = []

    random.seed(42)
    track_keys = list(tracks.keys())
    random.shuffle(track_keys)

    for track_key in track_keys:
        if demo_count >= args.num_demos:
            break

        track_data = tracks[track_key]
        n_chunks = track_data["num_chunks"]

        for chunk_idx in range(min(3, n_chunks)):  # Up to 3 chunks per track
            if demo_count >= args.num_demos:
                break

            # Pick a random stem pair
            stem1_name, stem2_name = random.choice(stem_pairs)

            demo_name = f"demo_{demo_count:02d}_{track_key[:20]}_chunk{chunk_idx}"
            demo_name = demo_name.replace(" ", "_").replace("/", "_")

            print(f"\nGenerating demo {demo_count + 1}/{args.num_demos}:")
            print(f"  Track: {track_key}")
            print(f"  Stems: {stem1_name} + {stem2_name}")
            print(f"  Chunk: {chunk_idx}")

            try:
                metrics = generate_demo(
                    model=model,
                    stem1_data=track_data["stems"][stem1_name],
                    stem2_data=track_data["stems"][stem2_name],
                    stem1_name=stem1_name,
                    stem2_name=stem2_name,
                    chunk_idx=chunk_idx,
                    output_dir=args.output,
                    demo_name=demo_name,
                    sample_rate=args.sample_rate,
                    device=device,
                    alpha=args.alpha,
                )

                print(f"  MixRate: {metrics['mix_rate']:.4f}")
                all_metrics.append(metrics)
                demo_count += 1

            except Exception as e:
                print(f"  Error: {e}")
                continue

    # Print summary
    print("\n" + "="*60)
    print("DEMO GENERATION COMPLETE")
    print("="*60)
    print(f"Generated {demo_count} demos in {args.output}/")
    print()

    if all_metrics:
        avg_rate = sum(m["mix_rate"] for m in all_metrics) / len(all_metrics)
        print(f"Average MixRate: {avg_rate:.4f}")
        print()
        print("For each demo, compare:")
        print("  - oracle_mix.wav: D(E(A+B)) - encode-decode the real mix")
        print("  - latent_mix.wav: D(E(A)+E(B)) - decode sum of latents")
        print()
        print("If they sound the same, the latent space is linear!")

    # Save metrics summary
    summary_path = os.path.join(args.output, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "demos": all_metrics,
            "average_mix_rate": sum(m["mix_rate"] for m in all_metrics) / len(all_metrics) if all_metrics else 0,
            "alpha": args.alpha,
            "checkpoint": args.checkpoint,
        }, f, indent=2)
    print(f"\nMetrics saved to {summary_path}")


if __name__ == "__main__":
    main()
