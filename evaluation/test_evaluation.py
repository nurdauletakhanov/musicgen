"""
Test set evaluation for NeurIPS paper.

Runs alpha sweep on MUSDB test split and reports MixRate metrics.

Usage:
    python -m evaluation.test_evaluation \
        --checkpoint checkpoints/musdb-phase2-mixing/best_model.pth

    # Compare Phase 1 vs Phase 2
    python -m evaluation.test_evaluation \
        --checkpoint checkpoints/musdb-phase1-recon/best_model.pth \
        --checkpoint2 checkpoints/musdb-phase2-mixing/best_model.pth
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
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


def load_test_stem_pairs(chunks_dir: str, num_samples: int = 100) -> List[Dict]:
    """
    Load stem pairs from MUSDB test split.

    Returns list of dicts with x1_stft, x1_wave, x2_stft, x2_wave.
    """
    index_path = os.path.join(chunks_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    # Get test tracks - look for test split with stem pairs
    test_tracks = index.get("test", {})
    if not test_tracks:
        raise RuntimeError("No test data found in index.json")

    stem_order = ["drums", "bass", "other", "vocals"]
    stem_pairs = [
        ("drums", "vocals"),
        ("drums", "bass"),
        ("bass", "vocals"),
        ("other", "vocals"),
        ("drums", "other"),
        ("bass", "other"),
    ]

    samples = []

    for track_key, track_info in test_tracks.items():
        if not isinstance(track_info, dict) or "stems" not in track_info:
            continue

        stems = track_info["stems"]
        n_chunks = track_info.get("num_chunks", 0)

        # Check we have all stems
        if not all(s in stems for s in stem_order):
            continue

        # Load all stem data for this track
        stem_data = {}
        for stem_name in stem_order:
            path = os.path.join(chunks_dir, "test", stems[stem_name])
            if not os.path.exists(path):
                continue
            data = torch.load(path, map_location="cpu", weights_only=True)
            stem_data[stem_name] = {
                "x_stft": data["x_stft"].float(),
                "x_wave": data["x_wave"].float(),
            }

        if len(stem_data) < 4:
            continue

        # Create stem pair samples
        for chunk_idx in range(min(n_chunks, 5)):  # Limit chunks per track
            for stem1_name, stem2_name in stem_pairs:
                if stem1_name not in stem_data or stem2_name not in stem_data:
                    continue
                samples.append({
                    "x1_stft": stem_data[stem1_name]["x_stft"][chunk_idx],
                    "x1_wave": stem_data[stem1_name]["x_wave"][chunk_idx],
                    "x2_stft": stem_data[stem2_name]["x_stft"][chunk_idx],
                    "x2_wave": stem_data[stem2_name]["x_wave"][chunk_idx],
                    "track": track_key,
                    "stems": f"{stem1_name}+{stem2_name}",
                })

    # Shuffle and limit
    random.seed(42)
    random.shuffle(samples)
    return samples[:num_samples]


def evaluate_alpha(
    model: torch.nn.Module,
    samples: List[Dict],
    alpha: float,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluate a single alpha value on test samples."""
    beta = 1.0 - alpha

    interp_l1_list = []
    interp_mr_list = []
    real_l1_list = []
    real_mr_list = []

    for sample in samples:
        x1_stft = sample["x1_stft"].unsqueeze(0).to(device)
        x1_wave = sample["x1_wave"].unsqueeze(0).to(device)
        x2_stft = sample["x2_stft"].unsqueeze(0).to(device)
        x2_wave = sample["x2_wave"].unsqueeze(0).to(device)

        if x1_wave.dim() == 2:
            x1_wave = x1_wave.unsqueeze(1)
            x2_wave = x2_wave.unsqueeze(1)

        with autocast("cuda", enabled=use_amp):
            z1 = model.encoder(x1_stft)
            z2 = model.encoder(x2_stft)

            x_mix_wave = alpha * x1_wave + beta * x2_wave

            # Oracle
            x_mix_stft = model._compute_stft_from_wave(x_mix_wave)
            z_real = model.encoder(x_mix_stft)
            x_real_recon = model.decoder(z_real)

            # Interpolation
            z_interp = alpha * z1 + beta * z2
            x_interp = model.decoder(z_interp)

            l1_real = F.l1_loss(x_real_recon, x_mix_wave).item()
            mr_real = model.mrstft_loss(x_real_recon, x_mix_wave).item()
            l1_interp = F.l1_loss(x_interp, x_mix_wave).item()
            mr_interp = model.mrstft_loss(x_interp, x_mix_wave).item()

        interp_l1_list.append(l1_interp)
        interp_mr_list.append(mr_interp)
        real_l1_list.append(l1_real)
        real_mr_list.append(mr_real)

    # Compute stats
    mix_l1_weight = getattr(model, 'mix_l1_weight', 1.0)
    mix_mrstft_weight = getattr(model, 'mix_mrstft_weight', 1.0)

    interp_totals = [mix_l1_weight * l1 + mix_mrstft_weight * mr
                     for l1, mr in zip(interp_l1_list, interp_mr_list)]
    real_totals = [mix_l1_weight * l1 + mix_mrstft_weight * mr
                   for l1, mr in zip(real_l1_list, real_mr_list)]

    rates = [i / (r + 1e-8) for i, r in zip(interp_totals, real_totals)]

    return {
        "MixReconInterp_mean": np.mean(interp_totals),
        "MixReconInterp_std": np.std(interp_totals),
        "MixReconReal_mean": np.mean(real_totals),
        "MixReconReal_std": np.std(real_totals),
        "MixRate_mean": np.mean(rates),
        "MixRate_std": np.std(rates),
        "MixRate_median": np.median(rates),
        "MixRate_p90": np.percentile(rates, 90),
        "num_samples": len(samples),
    }


def run_test_evaluation(
    model: torch.nn.Module,
    chunks_dir: str,
    alphas: List[float] = None,
    num_samples: int = 100,
    device: torch.device = None,
) -> Dict[float, Dict[str, float]]:
    """
    Run full test set evaluation with alpha sweep.

    Returns dict mapping alpha -> metrics.
    """
    if alphas is None:
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Load test samples
    print(f"Loading test samples from {chunks_dir}...")
    samples = load_test_stem_pairs(chunks_dir, num_samples)
    print(f"Loaded {len(samples)} test stem pairs")

    results = {}
    with torch.no_grad():
        for alpha in alphas:
            print(f"  Evaluating alpha={alpha}...")
            results[alpha] = evaluate_alpha(model, samples, alpha, device)

    return results


def print_results(results: Dict[float, Dict[str, float]], name: str = "Model"):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Test Set Evaluation: {name}")
    print(f"{'='*60}")
    print(f"{'Alpha':<8} {'MixRate':<12} {'Std':<10} {'Median':<10} {'P90':<10}")
    print(f"{'-'*60}")

    for alpha in sorted(results.keys()):
        m = results[alpha]
        print(f"{alpha:<8.1f} {m['MixRate_mean']:<12.4f} {m['MixRate_std']:<10.4f} "
              f"{m['MixRate_median']:<10.4f} {m['MixRate_p90']:<10.4f}")

    # Compute overall mean MixRate
    mean_rates = [results[a]['MixRate_mean'] for a in results]
    print(f"{'-'*60}")
    print(f"{'Overall':<8} {np.mean(mean_rates):<12.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test set evaluation for NeurIPS paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Phase 2 checkpoint"
    )
    parser.add_argument(
        "--checkpoint2",
        type=str,
        default=None,
        help="Optional: Path to Phase 1 checkpoint for comparison"
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="./musdb-chunks-stft",
        help="Directory containing preprocessed MUSDB chunks"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of test samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save results to JSON file"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Evaluate main checkpoint
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    results = run_test_evaluation(
        model, args.chunks_dir, alphas, args.num_samples, device
    )
    print_results(results, name=os.path.basename(os.path.dirname(args.checkpoint)))

    all_results = {"phase2": results}

    # Compare with Phase 1 if provided
    if args.checkpoint2:
        print(f"\nLoading comparison model from {args.checkpoint2}...")
        model2 = load_model(args.checkpoint2, device)
        results2 = run_test_evaluation(
            model2, args.chunks_dir, alphas, args.num_samples, device
        )
        print_results(results2, name=os.path.basename(os.path.dirname(args.checkpoint2)))
        all_results["phase1"] = results2

        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON: Phase 1 vs Phase 2")
        print("="*60)
        print(f"{'Alpha':<8} {'Phase1 MixRate':<16} {'Phase2 MixRate':<16} {'Improvement':<12}")
        print("-"*60)
        for alpha in alphas:
            r1 = results2[alpha]['MixRate_mean']
            r2 = results[alpha]['MixRate_mean']
            improvement = (r1 - r2) / r1 * 100 if r2 < r1 else (r2 - r1) / r2 * -100
            print(f"{alpha:<8.1f} {r1:<16.4f} {r2:<16.4f} {improvement:+.1f}%")

    # Save results if requested
    if args.output:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        with open(args.output, "w") as f:
            json.dump(convert(all_results), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
