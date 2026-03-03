"""
Generate publication-ready figures for NeurIPS paper.

Creates:
1. Alpha-sweep curve (MixRate vs alpha) — Phase 1, Phase 2, Encodec, DAC
2. Training loss curves (Phase 1 -> Phase 2) with selected checkpoint marker
3. MixRate over epochs with selected checkpoint marker
4. Reconstruction component curves (L1, MR-STFT)

Usage:
    python -m evaluation.plot_results \
        --phase1 checkpoints/musdb-phase1-recon \
        --phase2 checkpoints/musdb-phase2-mixing \
        --output results/figures/

    # With baseline comparison data:
    python -m evaluation.plot_results \
        --phase1 checkpoints/musdb-phase1-recon \
        --phase2 checkpoints/musdb-phase2-mixing \
        --baselines results/baseline_comparison.json \
        --output paper/figures/
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from evaluation import SELECTED_EPOCH

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
    """Extract metrics from all log files in checkpoint directory."""
    log_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".log")]
    if not log_files:
        return {"epochs": [], "alpha_sweeps": {}}

    # Merge all log files (in case training was resumed)
    all_epochs = []
    all_sweeps = {}
    for log_file in sorted(log_files):
        log_path = os.path.join(checkpoint_dir, log_file)
        parsed = parse_log_file(log_path)
        all_epochs.extend(parsed["epochs"])
        all_sweeps.update(parsed["alpha_sweeps"])

    return {
        "epochs": all_epochs,
        "alpha_sweeps": all_sweeps,
    }


def plot_alpha_sweep(
    phase1_metrics: Dict,
    phase2_metrics: Dict,
    output_path: str,
    baselines: Dict = None,
    test_eval_json: str = None,
):
    """
    Plot MixRate vs alpha curve for all models.

    Shows Phase 1, Phase 2 (selected epoch), and optionally Encodec/DAC baselines.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Phase 1 alpha sweep (from training log)
    p1_sweeps = phase1_metrics.get("alpha_sweeps", {})
    if p1_sweeps:
        p1_epoch = max(p1_sweeps.keys())
        p1_data = p1_sweeps[p1_epoch]
        p1_alphas = sorted(p1_data.keys())
        p1_rates = [p1_data[a]["mix_rate"] for a in p1_alphas]
        ax.plot(p1_alphas, p1_rates, 's--', linewidth=2, markersize=7,
                color='#2E86AB', label='Phase 1 (recon only)')

    # Phase 2 alpha sweep: prefer test evaluation JSON if provided
    p2_plotted = False
    if test_eval_json and os.path.exists(test_eval_json):
        with open(test_eval_json) as f:
            test_data = json.load(f)
        p2_test = test_data.get("phase2", {})
        if p2_test:
            p2_alphas = sorted([float(a) for a in p2_test.keys()])
            p2_rates = [p2_test[str(a)]["MixRate_mean"] for a in p2_alphas]
            ax.plot(p2_alphas, p2_rates, 'o-', linewidth=2, markersize=8,
                    color='#A23B72', label=f'Ours (Phase 2, epoch {SELECTED_EPOCH})')
            p2_plotted = True

    if not p2_plotted:
        # Fall back to log data
        p2_sweeps = phase2_metrics.get("alpha_sweeps", {})
        if p2_sweeps:
            if SELECTED_EPOCH in p2_sweeps:
                p2_epoch = SELECTED_EPOCH
            else:
                p2_epoch = min(p2_sweeps.keys(), key=lambda e: abs(e - SELECTED_EPOCH))
            p2_data = p2_sweeps[p2_epoch]
            p2_alphas = sorted(p2_data.keys())
            p2_rates = [p2_data[a]["mix_rate"] for a in p2_alphas]
            ax.plot(p2_alphas, p2_rates, 'o-', linewidth=2, markersize=8,
                    color='#A23B72', label=f'Ours (Phase 2, epoch {p2_epoch})')

    # Baselines
    if baselines:
        baseline_styles = {
            "encodec_48khz": ("^:", '#E8A838', 'EnCodec 48kHz'),
            "dac_44khz": ("D:", '#4CAF50', 'DAC 44kHz'),
        }
        for model_key, (style, color, label) in baseline_styles.items():
            if model_key in baselines:
                b_data = baselines[model_key]
                b_alphas = sorted([float(a) for a in b_data.keys()])
                b_rates = [b_data[str(a) if str(a) in b_data else a]["MixRate_mean"] for a in b_alphas]
                ax.plot(b_alphas, b_rates, style, linewidth=2, markersize=7,
                        color=color, label=label)

    # Ideal line
    ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Ideal (MixRate=1)')

    ax.set_xlabel(r'Mixing ratio $\alpha$')
    ax.set_ylabel('MixRate')
    ax.set_xlim(0, 1)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.legend(loc='best', framealpha=0.9, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved alpha sweep plot to {output_path}")


def plot_training_curves(
    phase1_metrics: Dict,
    phase2_metrics: Dict,
    output_path: str,
):
    """Plot training loss curves across Phase 1 and Phase 2."""
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

    # Mark selected checkpoint
    selected = [e for e in p2_epochs if e.get("epoch") == SELECTED_EPOCH]
    if selected:
        sel = selected[0]
        ax.plot(sel["epoch"], sel["val_loss"], '*', markersize=15, color='#E8A838',
                markeredgecolor='black', markeredgewidth=0.5, zorder=5,
                label=f'Selected (epoch {SELECTED_EPOCH})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved training curves to {output_path}")


def plot_mixrate_over_epochs(
    phase2_metrics: Dict,
    output_path: str,
):
    """Plot MixRate improvement over Phase 2 training epochs."""
    epochs_data = phase2_metrics.get("epochs", [])

    epochs_with_rate = [e for e in epochs_data if "val_mix_rate" in e]
    if not epochs_with_rate:
        print("No MixRate data for plotting")
        return

    epochs = [e["epoch"] for e in epochs_with_rate]
    rates = [e["val_mix_rate"] for e in epochs_with_rate]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epochs, rates, 'o-', linewidth=2, markersize=4, color='#2E86AB')
    ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Ideal (MixRate=1)')

    # Mark selected checkpoint
    selected = [e for e in epochs_with_rate if e.get("epoch") == SELECTED_EPOCH]
    if selected:
        sel = selected[0]
        ax.plot(sel["epoch"], sel["val_mix_rate"], '*', markersize=18, color='#E8A838',
                markeredgecolor='black', markeredgewidth=0.5, zorder=5,
                label=f'Selected (epoch {SELECTED_EPOCH}, MixRate={sel["val_mix_rate"]:.3f})')

    # Shade the "sweet spot" region near 1.0
    ax.axhspan(0.99, 1.01, alpha=0.1, color='green', label='Near-ideal zone')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MixRate')
    ax.set_ylim(0.92, 1.05)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved MixRate over epochs to {output_path}")


def plot_reconstruction_components(
    phase1_metrics: Dict,
    phase2_metrics: Dict,
    output_path: str,
):
    """Plot L1 and MR-STFT loss components over training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for phase_metrics, phase_num, color in [
        (phase1_metrics, 1, '#2E86AB'),
        (phase2_metrics, 2, '#A23B72'),
    ]:
        ep_data = [e for e in phase_metrics.get("epochs", []) if "val_l1" in e and "val_mrstft" in e]
        if not ep_data:
            continue

        epochs = [e["epoch"] for e in ep_data]
        l1_vals = [e["val_l1"] for e in ep_data]
        mr_vals = [e["val_mrstft"] for e in ep_data]

        ax1.plot(epochs, l1_vals, '-', linewidth=2, color=color, label=f'Phase {phase_num}')
        ax2.plot(epochs, mr_vals, '-', linewidth=2, color=color, label=f'Phase {phase_num}')

    # Mark selected checkpoint on both
    p2_epochs = phase2_metrics.get("epochs", [])
    selected = [e for e in p2_epochs if e.get("epoch") == SELECTED_EPOCH]
    if selected:
        sel = selected[0]
        if "val_l1" in sel:
            ax1.plot(sel["epoch"], sel["val_l1"], '*', markersize=12, color='#E8A838',
                     markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        if "val_mrstft" in sel:
            ax2.plot(sel["epoch"], sel["val_mrstft"], '*', markersize=12, color='#E8A838',
                     markeredgecolor='black', markeredgewidth=0.5, zorder=5)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('L1 Loss')
    ax1.set_title('Waveform L1 Loss')
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=8)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MR-STFT Loss')
    ax2.set_title('Multi-Resolution STFT Loss')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)

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
        "--baselines",
        type=str,
        default=None,
        help="Path to baseline_comparison.json (Encodec/DAC results)"
    )
    parser.add_argument(
        "--test-eval-json",
        type=str,
        default=None,
        help="Path to test evaluation JSON for Phase 2 alpha sweep data"
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

    # Extract metrics from logs
    print(f"Parsing Phase 1 logs from {args.phase1}...")
    phase1_metrics = get_checkpoint_metrics(args.phase1)
    print(f"  Found {len(phase1_metrics.get('epochs', []))} epochs")

    print(f"Parsing Phase 2 logs from {args.phase2}...")
    phase2_metrics = get_checkpoint_metrics(args.phase2)
    print(f"  Found {len(phase2_metrics.get('epochs', []))} epochs")
    print(f"  Found {len(phase2_metrics.get('alpha_sweeps', {}))} alpha sweeps")

    # Load baselines if provided
    baselines = None
    if args.baselines and os.path.exists(args.baselines):
        print(f"Loading baselines from {args.baselines}...")
        with open(args.baselines, "r") as f:
            baselines = json.load(f)

    # Generate figures
    print("\nGenerating figures...")

    plot_alpha_sweep(
        phase1_metrics,
        phase2_metrics,
        os.path.join(args.output, f"alpha_sweep.{args.format}"),
        baselines=baselines,
        test_eval_json=args.test_eval_json,
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
