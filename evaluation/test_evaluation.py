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
from typing import Dict, List, Optional

import numpy as np
import torch

from evaluation.utils import load_model, load_test_stem_pairs, evaluate_alpha_on_pairs


def evaluate_alpha(
    model: torch.nn.Module,
    samples: List[Dict],
    alpha: float,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluate a single alpha value on test samples with per-sample statistics."""
    return evaluate_alpha_on_pairs(
        model=model, pairs=samples, alpha=alpha,
        device=device, use_amp=use_amp, per_sample_stats=True,
    )


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
        default="./musdb-chunks-stft-v2",
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
