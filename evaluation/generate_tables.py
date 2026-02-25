"""
Generate LaTeX tables for NeurIPS paper.

Extracts metrics from training logs and checkpoints to create
publication-ready ablation tables.

Usage:
    python -m evaluation.generate_tables \
        --phase1 checkpoints/musdb-phase1-recon \
        --phase2 checkpoints/musdb-phase2-mixing \
        --output results/tables/
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch


def parse_log_file(log_path: str) -> Dict:
    """
    Parse training log file to extract metrics.

    Returns dict with:
        - epochs: list of epoch metrics
        - best_epoch: epoch with best val loss
        - final_epoch: last epoch metrics
        - alpha_sweep: dict of alpha sweep results (if present)
    """
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

        # Training metrics
        train_match = re.match(
            r"Train - Loss: ([\d.]+) \| Recon: ([\d.]+) \(L1: ([\d.]+), MR: ([\d.]+)\) "
            r"\| MixInterp: ([\d.]+), MixReal: ([\d.]+) \| Rate: ([\d.]+), Gap: ([-\d.]+)",
            line
        )
        if train_match:
            epoch_data["train_loss"] = float(train_match.group(1))
            epoch_data["train_recon"] = float(train_match.group(2))
            epoch_data["train_l1"] = float(train_match.group(3))
            epoch_data["train_mrstft"] = float(train_match.group(4))
            epoch_data["train_mix_interp"] = float(train_match.group(5))
            epoch_data["train_mix_real"] = float(train_match.group(6))
            epoch_data["train_mix_rate"] = float(train_match.group(7))
            continue

        # Validation metrics
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
            epoch_data["val_mix_interp"] = float(val_match.group(5))
            epoch_data["val_mix_real"] = float(val_match.group(6))
            epoch_data["val_mix_rate"] = float(val_match.group(7))
            continue

        # Phase 1 style (no mixing metrics)
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
            alpha_match = re.match(r"\s+alpha=([\d.]+): MixReconInterp=([\d.]+), MixRate=([\d.]+)", line)
            if alpha_match:
                alpha = float(alpha_match.group(1))
                alpha_sweeps[alpha_sweep_epoch][alpha] = {
                    "mix_recon_interp": float(alpha_match.group(2)),
                    "mix_rate": float(alpha_match.group(3)),
                }

        # Best val loss line
        if "Best val loss:" in line:
            match = re.search(r"Best val loss: ([\d.]+)", line)
            if match:
                epoch_data["best_val_loss"] = float(match.group(1))

    # Add last epoch
    if current_epoch is not None and epoch_data:
        epochs.append(epoch_data)

    # Find best epoch
    best_epoch = None
    best_loss = float("inf")
    for e in epochs:
        if e.get("val_loss", float("inf")) < best_loss:
            best_loss = e["val_loss"]
            best_epoch = e

    return {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "final_epoch": epochs[-1] if epochs else None,
        "alpha_sweeps": alpha_sweeps,
    }


def get_checkpoint_metrics(checkpoint_dir: str) -> Dict:
    """
    Extract metrics from a checkpoint directory.

    Looks for log files and parses them.
    """
    log_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".log")]
    if not log_files:
        return {}

    # Use most recent log file
    log_files.sort(reverse=True)
    log_path = os.path.join(checkpoint_dir, log_files[0])

    return parse_log_file(log_path)


def generate_main_table(
    phase1_metrics: Dict,
    phase2_metrics: Dict,
    output_path: str,
):
    """Generate main ablation table comparing Phase 1 and Phase 2."""

    p1_best = phase1_metrics.get("best_epoch", {})
    p2_best = phase2_metrics.get("best_epoch", {})

    # Get alpha sweep from final epoch if available
    p2_sweeps = phase2_metrics.get("alpha_sweeps", {})
    p2_final_sweep = {}
    if p2_sweeps:
        last_sweep_epoch = max(p2_sweeps.keys())
        p2_final_sweep = p2_sweeps[last_sweep_epoch]

    latex = r"""\begin{table}[t]
\centering
\caption{Ablation: Effect of decode-mixing loss on reconstruction quality and latent linearity.
Phase 2 adds the mixing equivariance constraint ($\lambda_\text{mix}=0.01$) to encourage
$D(E(A) + E(B)) \approx A + B$.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Val Loss} & \textbf{L1} & \textbf{MR-STFT} & \textbf{MixRate} \\
\midrule
"""

    # Phase 1 row
    p1_loss = p1_best.get("val_loss", "-")
    p1_l1 = p1_best.get("val_l1", "-")
    p1_mr = p1_best.get("val_mrstft", "-")
    p1_rate = p1_best.get("val_mix_rate", "-")

    if p1_loss != "-":
        p1_loss = f"{p1_loss:.3f}"
    if p1_l1 != "-":
        p1_l1 = f"{p1_l1:.3f}"
    if p1_mr != "-":
        p1_mr = f"{p1_mr:.3f}"
    if p1_rate != "-":
        p1_rate = f"{p1_rate:.3f}"
    else:
        p1_rate = "N/A"

    latex += f"Phase 1 (recon only) & {p1_loss} & {p1_l1} & {p1_mr} & {p1_rate} \\\\\n"

    # Phase 2 row
    p2_loss = p2_best.get("val_loss", "-")
    p2_l1 = p2_best.get("val_l1", "-")
    p2_mr = p2_best.get("val_mrstft", "-")
    p2_rate = p2_best.get("val_mix_rate", "-")

    if p2_loss != "-":
        p2_loss = f"{p2_loss:.3f}"
    if p2_l1 != "-":
        p2_l1 = f"{p2_l1:.3f}"
    if p2_mr != "-":
        p2_mr = f"{p2_mr:.3f}"
    if p2_rate != "-":
        p2_rate = f"\\textbf{{{p2_rate:.3f}}}"

    latex += f"Phase 2 (+ mixing loss) & {p2_loss} & {p2_l1} & {p2_mr} & {p2_rate} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved main table to {output_path}")

    return latex


def generate_alpha_table(
    phase2_metrics: Dict,
    output_path: str,
):
    """Generate alpha sweep table."""

    sweeps = phase2_metrics.get("alpha_sweeps", {})
    if not sweeps:
        print("No alpha sweep data found")
        return ""

    # Use the final sweep
    final_epoch = max(sweeps.keys())
    sweep_data = sweeps[final_epoch]

    latex = r"""\begin{table}[t]
\centering
\caption{MixRate across different mixing ratios $\alpha$.
Values close to 1.0 indicate the latent space supports linear mixing.
Evaluated at epoch """ + str(final_epoch) + r""".}
\label{tab:alpha_sweep}
\begin{tabular}{lccccc}
\toprule
$\alpha$ & 0.1 & 0.3 & 0.5 & 0.7 & 0.9 \\
\midrule
MixRate & """

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    rates = []
    for alpha in alphas:
        if alpha in sweep_data:
            rates.append(f"{sweep_data[alpha]['mix_rate']:.3f}")
        else:
            rates.append("-")

    latex += " & ".join(rates)

    latex += r""" \\
\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved alpha sweep table to {output_path}")

    return latex


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for NeurIPS paper",
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
        default="results/tables",
        help="Output directory for LaTeX files"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Extract metrics
    print(f"Parsing Phase 1 logs from {args.phase1}...")
    phase1_metrics = get_checkpoint_metrics(args.phase1)

    print(f"Parsing Phase 2 logs from {args.phase2}...")
    phase2_metrics = get_checkpoint_metrics(args.phase2)

    # Print summary
    if phase1_metrics.get("best_epoch"):
        p1 = phase1_metrics["best_epoch"]
        print(f"\nPhase 1 best: epoch {p1.get('epoch')}, val_loss={p1.get('val_loss'):.4f}")

    if phase2_metrics.get("best_epoch"):
        p2 = phase2_metrics["best_epoch"]
        print(f"Phase 2 best: epoch {p2.get('epoch')}, val_loss={p2.get('val_loss'):.4f}, "
              f"MixRate={p2.get('val_mix_rate', 'N/A')}")

    # Generate tables
    print("\nGenerating tables...")
    generate_main_table(
        phase1_metrics,
        phase2_metrics,
        os.path.join(args.output, "main_results.tex"),
    )

    generate_alpha_table(
        phase2_metrics,
        os.path.join(args.output, "alpha_sweep.tex"),
    )

    print(f"\nTables saved to {args.output}/")


if __name__ == "__main__":
    main()
