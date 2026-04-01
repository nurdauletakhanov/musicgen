"""
Batch evaluation of all lambda_mix sweep checkpoints for ISMIR 2026 paper.

Runs alpha sweep on each model and saves results to a single JSON file.

Usage:
    python -m evaluation.sweep_evaluation
    python -m evaluation.sweep_evaluation --output results/sweep_evaluation.json
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluation.utils import load_model, load_test_stem_pairs, evaluate_alpha_on_pairs, si_sdr


SWEEP_CHECKPOINTS = {
    "phase1": "checkpoints/comp-24x-v6/best_model.pth",
    "mix0.005": "checkpoints/comp-24x-v6-phase2-mix0.005/best_model.pth",
    "mix0.01": "checkpoints/comp-24x-v6-phase2-mix0.01/best_model.pth",
    "mix0.05": "checkpoints/comp-24x-v6-phase2-mix0.05/best_model.pth",
    "mix0.1": "checkpoints/comp-24x-v6-phase2-mix0.1/best_model.pth",
    "mix0.5": "checkpoints/comp-24x-v6-phase2-mix0.5/best_model.pth",
    "mix1.0": "checkpoints/comp-24x-v6-phase2-mix1.0/best_model.pth",
    "mix2.0": "checkpoints/comp-24x-v6-phase2-mix2.0/best_model.pth",
    "final": "checkpoints/comp-24x-v6-phase2-final/best_model.pth",
}

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj


def evaluate_single_model(name, checkpoint_path, pairs, alphas, device):
    """Evaluate one model on all alpha values."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name} ({checkpoint_path})")
    print(f"{'='*60}")

    if not os.path.exists(checkpoint_path):
        print(f"  SKIPPED: checkpoint not found")
        return None

    model = load_model(checkpoint_path, device)

    results = {}
    for alpha in alphas:
        print(f"  alpha={alpha}...", end=" ", flush=True)
        metrics = evaluate_alpha_on_pairs(
            model=model, pairs=pairs, alpha=alpha,
            device=device, use_amp=True, per_sample_stats=True,
        )
        results[alpha] = metrics
        print(f"MixRate={metrics['MixRate_mean']:.4f} (std={metrics['MixRate_std']:.4f})")

    # Summary
    mean_rate = np.mean([results[a]['MixRate_mean'] for a in alphas])
    print(f"  Overall mean MixRate: {mean_rate:.4f}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return results


def evaluate_reconstruction(name, checkpoint_path, pairs, device):
    """Evaluate reconstruction quality on single stems."""
    from evaluation.utils import load_test_singles

    if not os.path.exists(checkpoint_path):
        return None

    model = load_model(checkpoint_path, device)
    singles = load_test_singles("./musdb-chunks-stft-5s", num_samples=200)

    l1_list = []
    mr_list = []
    sisdr_list = []

    for sample in singles:
        x_stft = sample["x_stft"].unsqueeze(0).to(device)
        x_wave = sample["x_wave"].unsqueeze(0).to(device)
        if x_wave.dim() == 2:
            x_wave = x_wave.unsqueeze(1)
        tgt = model.decoder.target_length
        if x_wave.size(-1) > tgt:
            x_wave = x_wave[..., :tgt]

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                z = model.encoder(x_stft)
                x_hat, _ = model.decoder(z)
                l1 = torch.nn.functional.l1_loss(x_hat, x_wave).item()
                mr = model.mrstft_loss(x_hat, x_wave).item()
                sisdr_val = si_sdr(x_hat.float(), x_wave.float())

        l1_list.append(l1)
        mr_list.append(mr)
        sisdr_list.append(sisdr_val)

    del model
    torch.cuda.empty_cache()

    return {
        "recon_l1": float(np.mean(l1_list)),
        "recon_mrstft": float(np.mean(mr_list)),
        "recon_total": float(np.mean(l1_list)) + float(np.mean(mr_list)),
        "si_sdr": float(np.mean(sisdr_list)),
        "si_sdr_std": float(np.std(sisdr_list)),
        "num_samples": len(singles),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of lambda_mix sweep checkpoints"
    )
    parser.add_argument(
        "--output", type=str, default="results/sweep_evaluation.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--chunks-dir", type=str, default="./musdb-chunks-stft-5s",
        help="MUSDB chunks directory",
    )
    parser.add_argument(
        "--num-samples", type=int, default=200,
        help="Number of test stem pairs",
    )
    parser.add_argument(
        "--skip-recon", action="store_true",
        help="Skip reconstruction evaluation (faster)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test data once
    print(f"Loading test stem pairs from {args.chunks_dir}...")
    pairs = load_test_stem_pairs(args.chunks_dir, args.num_samples)
    print(f"Loaded {len(pairs)} test stem pairs")

    all_results = {}

    for name, ckpt_path in SWEEP_CHECKPOINTS.items():
        # Alpha sweep (MixRate)
        alpha_results = evaluate_single_model(
            name, ckpt_path, pairs, ALPHAS, device
        )
        if alpha_results is None:
            continue

        entry = {"alpha_sweep": alpha_results}

        # Reconstruction quality
        if not args.skip_recon:
            print(f"  Evaluating reconstruction quality...")
            recon = evaluate_reconstruction(name, ckpt_path, pairs, device)
            if recon:
                entry["reconstruction"] = recon
                print(f"  Recon: L1={recon['recon_l1']:.4f}, "
                      f"MRSTFT={recon['recon_mrstft']:.4f}, "
                      f"Total={recon['recon_total']:.4f}")

        all_results[name] = entry

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: MixRate @ alpha=0.5")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'MixRate':<10} {'Std':<10} {'Median':<10} {'Recon':<10} {'SI-SDR':<10}")
    print(f"{'-'*80}")
    for name, entry in all_results.items():
        alpha_data = entry["alpha_sweep"].get(0.5, {})
        rate = alpha_data.get("MixRate_mean", float("nan"))
        std = alpha_data.get("MixRate_std", float("nan"))
        median = alpha_data.get("MixRate_median", float("nan"))
        recon = entry.get("reconstruction", {}).get("recon_total", float("nan"))
        sisdr = entry.get("reconstruction", {}).get("si_sdr", float("nan"))
        print(f"{name:<15} {rate:<10.4f} {std:<10.4f} {median:<10.4f} {recon:<10.4f} {sisdr:<10.2f}")


if __name__ == "__main__":
    main()
