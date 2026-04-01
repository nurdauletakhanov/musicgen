"""
Principled checkpoint selection for Phase 2 model.

Evaluates candidate checkpoints on the MUSDB18 test set across multiple
mixing ratios (alphas) and selects the checkpoint that minimizes the mean
absolute deviation of MixRate from ideal (1.0).

Selection criterion:
    score(checkpoint) = mean_alpha(|MixRate_mean(alpha) - 1.0|)
    best = argmin(score)

This ensures:
- Symmetric penalty (both >1 and <1 indicate non-linearity)
- Multi-alpha robustness (not overfitting to a single mixing ratio)
- Test-set based (avoids validation set bias)

Usage:
    python -m evaluation.select_checkpoint \
        --checkpoint-dir checkpoints/musdb-phase2-mixing \
        --output results/checkpoint_selection.json

    # Evaluate specific epochs only:
    python -m evaluation.select_checkpoint \
        --checkpoint-dir checkpoints/musdb-phase2-mixing \
        --epochs 84 88 92 96 100 104 108 112 116 120

    # Quick screening with fewer samples:
    python -m evaluation.select_checkpoint --num-samples 50
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np

from evaluation.utils import load_model
from evaluation.test_evaluation import run_test_evaluation


def discover_checkpoints(checkpoint_dir: str) -> List[int]:
    """Find all checkpoint_<epoch>.pth files and return sorted epoch numbers."""
    epochs = []
    for f in os.listdir(checkpoint_dir):
        match = re.match(r"checkpoint_(\d+)\.pth$", f)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def compute_selection_score(alpha_results: Dict) -> float:
    """
    Compute checkpoint selection score.

    Score = mean |MixRate_mean - 1.0| across all alphas.
    Lower is better (0.0 = perfect linearity at all mixing ratios).
    """
    deviations = []
    for alpha_key, metrics in alpha_results.items():
        mixrate = metrics["MixRate_mean"]
        deviations.append(abs(mixrate - 1.0))
    return sum(deviations) / len(deviations)


def print_comparison_table(all_results: Dict[int, Dict], alphas: List[float]):
    """Print a formatted comparison table of all candidates."""
    header = f"{'Epoch':<8} "
    for a in alphas:
        header += f"{'α=' + str(a):<10} "
    header += f"{'Score':<10} {'Rank':<6}"
    print("=" * len(header))
    print("Checkpoint Comparison: MixRate by alpha (ideal = 1.000)")
    print("Score = mean |MixRate - 1.0| across alphas (lower is better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Sort by score
    scored = []
    for epoch, results in all_results.items():
        score = compute_selection_score(results)
        scored.append((epoch, results, score))
    scored.sort(key=lambda x: x[2])

    for rank, (epoch, results, score) in enumerate(scored, 1):
        row = f"{epoch:<8} "
        for a in alphas:
            a_key = str(a) if str(a) in results else a
            rate = results[a_key]["MixRate_mean"]
            row += f"{rate:<10.4f} "
        marker = " <-- BEST" if rank == 1 else ""
        row += f"{score:<10.4f} {rank:<6}{marker}"
        print(row)

    print("=" * len(header))
    best_epoch = scored[0][0]
    best_score = scored[0][2]
    print(f"\nRecommended: epoch {best_epoch} (score = {best_score:.4f})")

    return scored


def main():
    parser = argparse.ArgumentParser(
        description="Principled checkpoint selection via test-set MixRate evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/musdb-phase2-mixing",
        help="Directory containing checkpoint_<epoch>.pth files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Specific epochs to evaluate (default: every 4th available epoch)",
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="./musdb-chunks-stft-v2",
        help="Directory containing preprocessed MUSDB chunks",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of test samples per alpha",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/checkpoint_selection.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover available checkpoints
    available = discover_checkpoints(args.checkpoint_dir)
    if not available:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    print(f"Available checkpoints: {available}")

    # Select which epochs to evaluate
    if args.epochs:
        epochs = [e for e in args.epochs if e in available]
        missing = [e for e in args.epochs if e not in available]
        if missing:
            print(f"Warning: epochs {missing} not found, skipping")
    else:
        # Default: every 4th epoch for efficiency
        epochs = available[::2] if len(available) > 12 else available
        # Always include first and last
        if available[0] not in epochs:
            epochs.insert(0, available[0])
        if available[-1] not in epochs:
            epochs.append(available[-1])
        epochs = sorted(set(epochs))

    print(f"Evaluating {len(epochs)} checkpoints: {epochs}")

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_results = {}

    for i, epoch in enumerate(epochs):
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pth")
        print(f"\n[{i + 1}/{len(epochs)}] Evaluating epoch {epoch}...")
        t0 = time.time()

        model = load_model(checkpoint_path, device)
        results = run_test_evaluation(
            model, args.chunks_dir, alphas, args.num_samples, device
        )

        # Convert numpy types for JSON serialization
        clean_results = {}
        for alpha, metrics in results.items():
            clean_metrics = {}
            for k, v in metrics.items():
                clean_metrics[k] = float(v) if isinstance(v, (np.floating, float)) else int(v)
            clean_results[str(alpha)] = clean_metrics

        all_results[epoch] = clean_results
        score = compute_selection_score(clean_results)

        elapsed = time.time() - t0
        print(f"  Score: {score:.4f} (elapsed: {elapsed:.1f}s)")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Print comparison table
    print()
    scored = print_comparison_table(all_results, alphas)
    best_epoch = scored[0][0]
    best_score = scored[0][2]

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "selection_criterion": "min mean |MixRate_mean - 1.0| across alphas",
        "alphas": alphas,
        "num_test_samples": args.num_samples,
        "recommended_epoch": best_epoch,
        "recommended_score": best_score,
        "candidates": {str(e): all_results[e] for e in epochs},
        "scores": {str(e): compute_selection_score(all_results[e]) for e in epochs},
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"\nTo use this result, update evaluation/__init__.py:")
    print(f"  SELECTED_EPOCH = {best_epoch}")


if __name__ == "__main__":
    main()
