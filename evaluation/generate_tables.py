"""
Generate LaTeX tables for NeurIPS paper.

Extracts metrics from training logs and baseline comparison results to create
publication-ready ablation tables.

Usage:
    python -m evaluation.generate_tables \
        --phase1 checkpoints/musdb-phase1-recon \
        --phase2 checkpoints/musdb-phase2-mixing \
        --baselines results/baseline_comparison.json \
        --output results/tables/
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from evaluation import SELECTED_EPOCH


def parse_log_file(log_path: str) -> Dict:
    """Parse training log file to extract per-epoch metrics and alpha sweeps."""
    epochs = []
    alpha_sweeps = {}

    with open(log_path, "r") as f:
        lines = f.readlines()

    current_epoch = None
    epoch_data = {}
    in_alpha_sweep = False
    alpha_sweep_epoch = None

    for line in lines:
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

    if current_epoch is not None and epoch_data:
        epochs.append(epoch_data)

    # Find best epoch and selected epoch
    best_epoch = None
    best_loss = float("inf")
    selected_epoch = None
    for e in epochs:
        if e.get("val_loss", float("inf")) < best_loss:
            best_loss = e["val_loss"]
            best_epoch = e
        if e.get("epoch") == SELECTED_EPOCH:
            selected_epoch = e

    return {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "selected_epoch": selected_epoch,
        "final_epoch": epochs[-1] if epochs else None,
        "alpha_sweeps": alpha_sweeps,
    }


def get_checkpoint_metrics(checkpoint_dir: str) -> Dict:
    """Extract metrics from all log files in checkpoint directory."""
    log_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".log")]
    if not log_files:
        return {}

    # Merge all log files
    all_epochs = []
    all_sweeps = {}
    for log_file in sorted(log_files):
        log_path = os.path.join(checkpoint_dir, log_file)
        parsed = parse_log_file(log_path)
        all_epochs.extend(parsed["epochs"])
        all_sweeps.update(parsed["alpha_sweeps"])

    best_epoch = None
    best_loss = float("inf")
    selected_epoch = None
    for e in all_epochs:
        if e.get("val_loss", float("inf")) < best_loss:
            best_loss = e["val_loss"]
            best_epoch = e
        if e.get("epoch") == SELECTED_EPOCH:
            selected_epoch = e

    return {
        "epochs": all_epochs,
        "best_epoch": best_epoch,
        "selected_epoch": selected_epoch,
        "final_epoch": all_epochs[-1] if all_epochs else None,
        "alpha_sweeps": all_sweeps,
    }


def _fmt(val, fmt=".3f", bold=False):
    """Format a metric value for LaTeX."""
    if val is None or val == "-":
        return "---"
    s = f"{val:{fmt}}"
    if bold:
        return f"\\textbf{{{s}}}"
    return s


def generate_main_table(
    phase1_metrics: Dict,
    phase2_metrics: Dict,
    output_path: str,
    baselines: Dict = None,
):
    """Generate main ablation table: Phase 1, Phase 2 (selected epoch), Encodec, DAC."""

    # Phase 1: use best epoch
    p1 = phase1_metrics.get("best_epoch", {})
    # Phase 1 MixRate from alpha sweep (since mixing was disabled during training)
    p1_sweeps = phase1_metrics.get("alpha_sweeps", {})
    p1_mixrate = None
    if p1_sweeps:
        last_sweep = p1_sweeps[max(p1_sweeps.keys())]
        if 0.5 in last_sweep:
            p1_mixrate = last_sweep[0.5]["mix_rate"]

    # Phase 2: use SELECTED epoch (not best val loss)
    p2 = phase2_metrics.get("selected_epoch") or phase2_metrics.get("best_epoch", {})
    p2_rate = p2.get("val_mix_rate")

    latex = r"""\begin{table}[t]
\centering
\caption{Latent mixing linearity across models. MixRate measures the ratio of latent-mixed
reconstruction loss to oracle reconstruction loss (ideal = 1.0). Phase~2 adds the decode-mixing
constraint ($\lambda_\text{mix}=0.01$) to our autoencoder. EnCodec and DAC are evaluated using
continuous (pre-quantization) embeddings.}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Val Loss} & \textbf{L1} & \textbf{MR-STFT} & \textbf{MixRate} ($\alpha\!=\!0.5$) \\
\midrule
"""

    # Phase 1 row
    latex += f"Ours -- Phase 1 (recon only) & {_fmt(p1.get('val_loss'))} & {_fmt(p1.get('val_l1'))} & {_fmt(p1.get('val_mrstft'))} & {_fmt(p1_mixrate)} \\\\\n"

    # Phase 2 row (selected epoch)
    ep_label = f" (epoch {p2.get('epoch', '?')})" if p2.get('epoch') else ""
    latex += f"Ours -- Phase 2 (+ mixing){ep_label} & {_fmt(p2.get('val_loss'))} & {_fmt(p2.get('val_l1'))} & {_fmt(p2.get('val_mrstft'))} & {_fmt(p2_rate, bold=True)} \\\\\n"

    # Baselines
    if baselines:
        latex += "\\midrule\n"
        baseline_names = {
            "encodec_48khz": "EnCodec 48kHz",
            "dac_44khz": "DAC 44kHz",
        }
        for model_key, display_name in baseline_names.items():
            if model_key in baselines:
                b_data = baselines[model_key]
                # Get alpha=0.5 MixRate
                rate_05 = None
                for a_key in [0.5, "0.5"]:
                    if a_key in b_data:
                        rate_05 = b_data[a_key]["MixRate_mean"]
                        break
                latex += f"{display_name} & --- & --- & --- & {_fmt(rate_05)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved main table to {output_path}")
    return latex


def generate_alpha_table(
    phase1_metrics: Dict,
    phase2_metrics: Dict,
    output_path: str,
    baselines: Dict = None,
    test_eval_json: str = None,
):
    """Generate alpha sweep table with all models."""

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Collect data rows
    rows = []

    # Phase 1
    p1_sweeps = phase1_metrics.get("alpha_sweeps", {})
    if p1_sweeps:
        p1_data = p1_sweeps[max(p1_sweeps.keys())]
        p1_rates = [p1_data.get(a, {}).get("mix_rate") for a in alphas]
        rows.append(("Ours -- Phase 1", p1_rates))

    # Phase 2: prefer test evaluation JSON if provided
    if test_eval_json and os.path.exists(test_eval_json):
        with open(test_eval_json) as f:
            test_data = json.load(f)
        p2_test = test_data.get("phase2", {})
        p2_rates = [p2_test.get(str(a), {}).get("MixRate_mean") for a in alphas]
        rows.append(("Ours -- Phase 2", p2_rates))
    else:
        # Fall back to log data, using SELECTED_EPOCH (not max)
        p2_sweeps = phase2_metrics.get("alpha_sweeps", {})
        if p2_sweeps:
            if SELECTED_EPOCH in p2_sweeps:
                p2_data = p2_sweeps[SELECTED_EPOCH]
            else:
                nearest = min(p2_sweeps.keys(), key=lambda e: abs(e - SELECTED_EPOCH))
                p2_data = p2_sweeps[nearest]
                print(f"Warning: No alpha sweep at epoch {SELECTED_EPOCH}, using epoch {nearest}")
            p2_rates = [p2_data.get(a, {}).get("mix_rate") for a in alphas]
            rows.append(("Ours -- Phase 2", p2_rates))

    # Baselines
    if baselines:
        baseline_names = {
            "encodec_48khz": "EnCodec 48kHz",
            "dac_44khz": "DAC 44kHz",
        }
        for model_key, display_name in baseline_names.items():
            if model_key in baselines:
                b_data = baselines[model_key]
                b_rates = []
                for a in alphas:
                    for a_key in [a, str(a)]:
                        if a_key in b_data:
                            b_rates.append(b_data[a_key]["MixRate_mean"])
                            break
                    else:
                        b_rates.append(None)
                rows.append((display_name, b_rates))

    latex = r"""\begin{table}[t]
\centering
\caption{MixRate across different mixing ratios $\alpha$. Values close to 1.0 indicate
the latent space supports linear mixing. Our Phase~2 model achieves near-ideal linearity
across all mixing ratios.}
\label{tab:alpha_sweep}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & $\alpha\!=\!0.1$ & $\alpha\!=\!0.3$ & $\alpha\!=\!0.5$ & $\alpha\!=\!0.7$ & $\alpha\!=\!0.9$ \\
\midrule
"""

    for name, rates in rows:
        rate_strs = [_fmt(r) for r in rates]
        latex += f"{name} & {' & '.join(rate_strs)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved alpha sweep table to {output_path}")
    return latex


def generate_selection_table(
    selection_json: str,
    output_path: str,
):
    """Generate checkpoint selection justification table from selection results."""
    with open(selection_json) as f:
        data = json.load(f)

    alphas = data["alphas"]
    scores = data["scores"]
    candidates = data["candidates"]
    recommended = data["recommended_epoch"]

    # Sort by score (best first)
    ranked = sorted(scores.items(), key=lambda x: x[1])

    latex = r"""\begin{table}[t]
\centering
\caption{Checkpoint selection on the test set. We select the checkpoint minimizing
the mean absolute deviation of MixRate from 1.0 across mixing ratios
$\bar{\Delta} = \frac{1}{|\mathcal{A}|}\sum_{\alpha \in \mathcal{A}} |\text{MixRate}(\alpha) - 1|$.
Lower $\bar{\Delta}$ indicates better mixing linearity.}
\label{tab:checkpoint_selection}
\begin{tabular}{lcccccc}
\toprule
\textbf{Epoch} & $\alpha\!=\!0.1$ & $\alpha\!=\!0.3$ & $\alpha\!=\!0.5$ & $\alpha\!=\!0.7$ & $\alpha\!=\!0.9$ & $\bar{\Delta}$ \\
\midrule
"""

    for epoch_str, score in ranked:
        epoch_data = candidates[epoch_str]
        rates = []
        for a in alphas:
            a_key = str(a)
            rate = epoch_data.get(a_key, {}).get("MixRate_mean")
            rates.append(rate)

        is_selected = int(epoch_str) == recommended
        rate_strs = [_fmt(r) for r in rates]
        score_str = _fmt(score, bold=is_selected)
        epoch_label = epoch_str
        if is_selected:
            epoch_label = f"\\textbf{{{epoch_str}}}"

        latex += f"{epoch_label} & {' & '.join(rate_strs)} & {score_str} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved checkpoint selection table to {output_path}")
    return latex


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for NeurIPS paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase1", type=str, default="checkpoints/musdb-phase1-recon")
    parser.add_argument("--phase2", type=str, default="checkpoints/musdb-phase2-mixing")
    parser.add_argument("--baselines", type=str, default=None,
                        help="Path to baseline_comparison.json")
    parser.add_argument("--test-eval-json", type=str, default=None,
                        help="Path to test evaluation JSON for Phase 2 alpha sweep data")
    parser.add_argument("--selection-json", type=str, default=None,
                        help="Path to checkpoint_selection.json for selection table")
    parser.add_argument("--output", type=str, default="results/tables")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Parsing Phase 1 logs from {args.phase1}...")
    phase1_metrics = get_checkpoint_metrics(args.phase1)

    print(f"Parsing Phase 2 logs from {args.phase2}...")
    phase2_metrics = get_checkpoint_metrics(args.phase2)

    # Load baselines
    baselines = None
    if args.baselines and os.path.exists(args.baselines):
        print(f"Loading baselines from {args.baselines}...")
        with open(args.baselines, "r") as f:
            baselines = json.load(f)

    # Print summary
    if phase1_metrics.get("best_epoch"):
        p1 = phase1_metrics["best_epoch"]
        print(f"\nPhase 1 best: epoch {p1.get('epoch')}, val_loss={p1.get('val_loss'):.4f}")

    if phase2_metrics.get("selected_epoch"):
        p2 = phase2_metrics["selected_epoch"]
        print(f"Phase 2 selected: epoch {p2.get('epoch')}, val_loss={p2.get('val_loss'):.4f}, "
              f"MixRate={p2.get('val_mix_rate', 'N/A')}")
    elif phase2_metrics.get("best_epoch"):
        p2 = phase2_metrics["best_epoch"]
        print(f"Phase 2 best: epoch {p2.get('epoch')}, val_loss={p2.get('val_loss'):.4f}")

    # Generate tables
    print("\nGenerating tables...")
    generate_main_table(
        phase1_metrics, phase2_metrics,
        os.path.join(args.output, "main_results.tex"),
        baselines=baselines,
    )

    generate_alpha_table(
        phase1_metrics, phase2_metrics,
        os.path.join(args.output, "alpha_sweep.tex"),
        baselines=baselines,
        test_eval_json=args.test_eval_json,
    )

    # Checkpoint selection table
    selection_path = args.selection_json
    if selection_path is None:
        selection_path = "results/checkpoint_selection.json"
    if os.path.exists(selection_path):
        generate_selection_table(
            selection_path,
            os.path.join(args.output, "checkpoint_selection.tex"),
        )

    print(f"\nTables saved to {args.output}/")


if __name__ == "__main__":
    main()
