"""
Generate publication-ready figures for NeurIPS paper.

Creates:
1. Alpha-sweep curve (MixRate vs alpha)
2. Training loss curves (Phase 1 -> Phase 2)
3. MixRate over epochs

Usage:
    python -m evaluation.plot_results \
        --phase1 checkpoints/musdb-phase1-recon \
        --phase2 checkpoints/musdb-phase2-mixing \
        --output results/figures/
"""

import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend for saving
matplotlib.use('Agg')

# Publication-ready style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def parse_log_file(log_path: str) -> Dict:
    """Parse training log file to extract metrics."""
    epochs = []
    alpha_sweeps = {}

    with open(log_path, "r") as f:
        lines = f.readlines()

    current_epoch = None
    epoch_data = {}
    in_alpha_sweep = False
    alpha_sweep_epoch = None

    for line in lines:
        # Epoch header
        epoch_match = re.match(r"=== Epoch (\d+)/(\d+) ===", line)
        if epoch_match:
            if current_epoch is not None and epoch_data:
                epochs.append(epoch_data)
            current_epoch = int(epoch_match.group(1))
            epoch_data = {"epoch": current_epoch}
            in_alpha_sweep = False
            continue

        # Validation metrics with mixing
        val_match = re.match(
            r"Val   - Loss: ([\d.]+) \| Recon: ([\d.]+) \(L1: ([\d.]+), MR: ([\d.]+)\) "
            r"\| MixInterp: ([\d.]+), MixReal: ([\d.]+) \| Rate: ([\d.]+), Gap: ([-\d.]+)",
            line
        )
        if val_match:
            epoch_data["val_loss"] = float(val_match.group(1))
            epoch_data["val_recon"] = float(val_match.group(2))
            epoch_data["val_l1"] = float(val_match.group(3))
            epoch_data["val_mrstft"] = float(val_match.group(4))
            epoch_data["val_mix_rate"] = float(val_match.group(7))
            continue

        # Validation metrics without mixing (Phase 1)
        val_simple_match = re.match(
            r"Val   - Loss: ([\d.]+) \| Recon: ([\d.]+) \(L1: ([\d.]+), MR: ([\d.]+)\)",
            line
        )
        if val_simple_match and "val_loss" not in epoch_data:
            epoch_data["val_loss"] = float(val_simple_match.group(1))
            epoch_data["val_recon"] = float(val_simple_match.group(2))
            epoch_data["val_l1"] = float(val_simple_match.group(3))
            epoch_data["val_mrstft"] = float(val_simple_match.group(4))
            continue

        # Alpha sweep
        if "Running alpha sweep evaluation" in line:
            in_alpha_sweep = True
            alpha_sweep_epoch = current_epoch
            if alpha_sweep_epoch not in alpha_sweeps:
                alpha_sweeps[alpha_sweep_epoch] = {}
            continue

        if in_alpha_sweep:
            alpha_match = re.match(
                r"\s+alpha=([\d.]+): MixReconInterp=([\d.]+), MixRate=([\d.]+)",
                line
            )
            if alpha_match:
                alpha = float(alpha_match.group(1))
                alpha_sweeps[alpha_sweep_epoch][alpha] = {
                    "mix_recon_interp": float(alpha_match.group(2)),
                    "mix_rate": float(alpha_match.group(3)),
                }

    # Add last epoch
    if current_epoch is not None and epoch_data:
        epochs.append(epoch_data)

    return {
        "epochs": epochs,
        "alpha_sweeps": alpha_sweeps,
    }


def get_checkpoint_metrics(checkpoint_dir: str) -> Dict:
    """Extract metrics from checkpoint directory."""
    log_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".log")]
    if not log_files:
        return {"epochs": [], "alpha_sweeps": {}}

    log_files.sort(reverse=True)
    log_path = os.path.join(checkpoint_dir, log_files[0])
    return parse_log_file(log_path)


def plot_alpha_sweep(
    phase2_metrics: Dict,
    output_path: str,
):
    """
    Plot MixRate vs alpha curve.

    Shows how linearity holds across different mixing ratios.
    """
    sweeps = phase2_metrics.get("alpha_sweeps", {})
    if not sweeps:
        print("No alpha sweep data for plotting")
        return

    # Get final sweep
    final_epoch = max(sweeps.keys())
    sweep_data = sweeps[final_epoch]

    alphas = sorted(sweep_data.keys())
    rates = [sweep_data[a]["mix_rate"] for a in alphas]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(alphas, rates, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.axhline(y=1.0, color='#A23B72', linestyle='--', linewidth=1.5, label='Ideal (MixRate=1)')

    ax.set_xlabel(r'Mixing ratio $\alpha$')
    ax.set_ylabel('MixRate')
    ax.set_title(f'Latent Mixing Linearity (Epoch {final_epoch})')

    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 1.1)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])

    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved alpha sweep plot to {output_path}")


def plot_training_curves(
    phase1_metrics: Dict,
    phase2_metrics: Dict,
    output_path: str,
):
    """
    Plot training loss curves across Phase 1 and Phase 2.

    Shows the transition from reconstruction-only to mixing loss.
    """
    p1_epochs = phase1_metrics.get("epochs", [])
    p2_epochs = phase2_metrics.get("epochs", [])

    if not p1_epochs and not p2_epochs:
        print("No training data for plotting")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Phase 1
    if p1_epochs:
        epochs_p1 = [e["epoch"] for e in p1_epochs if "val_loss" in e]
        losses_p1 = [e["val_loss"] for e in p1_epochs if "val_loss" in e]
        ax.plot(epochs_p1, losses_p1, '-', linewidth=2, color='#2E86AB',
                label='Phase 1 (recon only)')

    # Phase 2
    if p2_epochs:
        epochs_p2 = [e["epoch"] for e in p2_epochs if "val_loss" in e]
        losses_p2 = [e["val_loss"] for e in p2_epochs if "val_loss" in e]
        ax.plot(epochs_p2, losses_p2, '-', linewidth=2, color='#A23B72',
                label='Phase 2 (+ mixing loss)')

    # Mark phase transition
    if p1_epochs and p2_epochs:
        transition_epoch = p1_epochs[-1]["epoch"]
        ax.axvline(x=transition_epoch, color='gray', linestyle=':', linewidth=1.5,
                   label=f'Phase transition (epoch {transition_epoch})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training Progression: Phase 1 → Phase 2')

    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved training curves to {output_path}")


def plot_mixrate_over_epochs(
    phase2_metrics: Dict,
    output_path: str,
):
    """
    Plot MixRate improvement over training epochs.

    Shows how mixing linearity improves during Phase 2.
    """
    epochs_data = phase2_metrics.get("epochs", [])

    epochs_with_rate = [e for e in epochs_data if "val_mix_rate" in e]
    if not epochs_with_rate:
        print("No MixRate data for plotting")
        return

    epochs = [e["epoch"] for e in epochs_with_rate]
    rates = [e["val_mix_rate"] for e in epochs_with_rate]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epochs, rates, 'o-', linewidth=2, markersize=4, color='#2E86AB')
    ax.axhline(y=1.0, color='#A23B72', linestyle='--', linewidth=1.5, label='Ideal (MixRate=1)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MixRate')
    ax.set_title('MixRate Improvement During Phase 2 Training')

    ax.set_ylim(0.9, 1.1)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved MixRate over epochs to {output_path}")


def plot_reconstruction_components(
    phase1_metrics: Dict,
    phase2_metrics: Dict,
    output_path: str,
):
    """
    Plot L1 and MR-STFT loss components over training.

    Shows reconstruction quality across both phases.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Combine data
    all_epochs = []
    all_l1 = []
    all_mr = []
    phases = []

    for e in phase1_metrics.get("epochs", []):
        if "val_l1" in e and "val_mrstft" in e:
            all_epochs.append(e["epoch"])
            all_l1.append(e["val_l1"])
            all_mr.append(e["val_mrstft"])
            phases.append(1)

    for e in phase2_metrics.get("epochs", []):
        if "val_l1" in e and "val_mrstft" in e:
            all_epochs.append(e["epoch"])
            all_l1.append(e["val_l1"])
            all_mr.append(e["val_mrstft"])
            phases.append(2)

    if not all_epochs:
        print("No reconstruction data for plotting")
        return

    # L1 plot
    p1_mask = [p == 1 for p in phases]
    p2_mask = [p == 2 for p in phases]

    epochs_p1 = [e for e, m in zip(all_epochs, p1_mask) if m]
    l1_p1 = [l for l, m in zip(all_l1, p1_mask) if m]
    mr_p1 = [m for m, mask in zip(all_mr, p1_mask) if mask]

    epochs_p2 = [e for e, m in zip(all_epochs, p2_mask) if m]
    l1_p2 = [l for l, m in zip(all_l1, p2_mask) if m]
    mr_p2 = [m for m, mask in zip(all_mr, p2_mask) if mask]

    ax1.plot(epochs_p1, l1_p1, '-', linewidth=2, color='#2E86AB', label='Phase 1')
    ax1.plot(epochs_p2, l1_p2, '-', linewidth=2, color='#A23B72', label='Phase 2')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('L1 Loss')
    ax1.set_title('Waveform L1 Loss')
    ax1.legend()

    ax2.plot(epochs_p1, mr_p1, '-', linewidth=2, color='#2E86AB', label='Phase 1')
    ax2.plot(epochs_p2, mr_p2, '-', linewidth=2, color='#A23B72', label='Phase 2')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MR-STFT Loss')
    ax2.set_title('Multi-Resolution STFT Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved reconstruction components to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for NeurIPS paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase1",
        type=str,
        default="checkpoints/musdb-phase1-recon",
        help="Path to Phase 1 checkpoint directory"
    )
    parser.add_argument(
        "--phase2",
        type=str,
        default="checkpoints/musdb-phase2-mixing",
        help="Path to Phase 2 checkpoint directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Extract metrics
    print(f"Parsing Phase 1 logs from {args.phase1}...")
    phase1_metrics = get_checkpoint_metrics(args.phase1)
    print(f"  Found {len(phase1_metrics.get('epochs', []))} epochs")

    print(f"Parsing Phase 2 logs from {args.phase2}...")
    phase2_metrics = get_checkpoint_metrics(args.phase2)
    print(f"  Found {len(phase2_metrics.get('epochs', []))} epochs")
    print(f"  Found {len(phase2_metrics.get('alpha_sweeps', {}))} alpha sweeps")

    # Generate figures
    print("\nGenerating figures...")

    plot_alpha_sweep(
        phase2_metrics,
        os.path.join(args.output, f"alpha_sweep.{args.format}"),
    )

    plot_training_curves(
        phase1_metrics,
        phase2_metrics,
        os.path.join(args.output, f"training_curves.{args.format}"),
    )

    plot_mixrate_over_epochs(
        phase2_metrics,
        os.path.join(args.output, f"mixrate_epochs.{args.format}"),
    )

    plot_reconstruction_components(
        phase1_metrics,
        phase2_metrics,
        os.path.join(args.output, f"reconstruction_components.{args.format}"),
    )

    print(f"\nAll figures saved to {args.output}/")


if __name__ == "__main__":
    main()
