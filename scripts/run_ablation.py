"""
Lambda_mix ablation automation.

Runs the full pipeline: train -> checkpoint selection -> test evaluation -> comparison table
for all lambda_mix variants.

Usage:
    # Run everything (train + select + evaluate + table)
    python scripts/run_ablation.py

    # Skip training, just evaluate and generate table
    python scripts/run_ablation.py --stages select evaluate table

    # Only specific lambda values
    python scripts/run_ablation.py --lambdas 0.001 0.05

    # Just regenerate comparison table from existing results
    python scripts/run_ablation.py --stages table

    # Force re-run (ignore existing outputs)
    python scripts/run_ablation.py --stages evaluate --force
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time


VARIANTS = [
    {
        "lambda_mix": 0.001,
        "config": "configs/experiments/musdb_phase2_mix0.001.yaml",
        "checkpoint_dir": "checkpoints/musdb-phase2-mix0.001",
    },
    {
        "lambda_mix": 0.01,
        "config": "configs/experiments/musdb_phase2_mixing.yaml",
        "checkpoint_dir": "checkpoints/musdb-phase2-mixing",
    },
    {
        "lambda_mix": 0.05,
        "config": "configs/experiments/musdb_phase2_mix0.05.yaml",
        "checkpoint_dir": "checkpoints/musdb-phase2-mix0.05",
    },
    {
        "lambda_mix": 0.1,
        "config": "configs/experiments/musdb_phase2_mix0.1.yaml",
        "checkpoint_dir": "checkpoints/musdb-phase2-mix0.1",
    },
    {
        "lambda_mix": 1.0,
        "config": "configs/experiments/musdb_phase2_mix1.0.yaml",
        "checkpoint_dir": "checkpoints/musdb-phase2-mix1.0",
    },
]

# Existing results for lambda=0.01 that can be reused
EXISTING_RESULTS = {
    "checkpoint_selection": "results/checkpoint_selection.json",
    "test_evaluation": "results/test_evaluation.json",
}


def fmt_time(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def lambda_key(lam: float) -> str:
    """Format lambda value for use in filenames."""
    return str(lam)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def detect_training_state(checkpoint_dir: str) -> str:
    """Detect training state: 'complete', 'partial', or 'none'."""
    if not os.path.isdir(checkpoint_dir):
        return "none"

    has_best = os.path.exists(os.path.join(checkpoint_dir, "best_model.pth"))
    checkpoints = find_checkpoint_files(checkpoint_dir)

    if not checkpoints:
        return "none"

    # Training runs 120 epochs total (80 Phase1 + 40 Phase2)
    # Complete if best_model exists and we have checkpoints near epoch 120
    max_epoch = max(checkpoints)
    if has_best and max_epoch >= 118:
        return "complete"
    elif checkpoints:
        return "partial"
    return "none"


def find_checkpoint_files(checkpoint_dir: str) -> list:
    """Find checkpoint epoch numbers in a directory."""
    epochs = []
    if not os.path.isdir(checkpoint_dir):
        return epochs
    for f in os.listdir(checkpoint_dir):
        match = re.match(r"checkpoint_(\d+)\.pth$", f)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint file for resuming interrupted training."""
    epochs = find_checkpoint_files(checkpoint_dir)
    if not epochs:
        return None
    return os.path.join(checkpoint_dir, f"checkpoint_{max(epochs)}.pth")


def parse_log_metrics(checkpoint_dir: str, target_epoch: int) -> dict | None:
    """Extract validation metrics for a specific epoch from training logs."""
    log_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".log")]
    if not log_files:
        return None

    for log_file in sorted(log_files):
        log_path = os.path.join(checkpoint_dir, log_file)
        with open(log_path, "r") as f:
            lines = f.readlines()

        current_epoch = None
        for line in lines:
            epoch_match = re.match(r"=== Epoch (\d+)/(\d+) ===", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            if current_epoch != target_epoch:
                continue

            # Phase 2 validation line (with mixing metrics)
            val_match = re.match(
                r"Val   - Loss: ([\d.]+) \| Recon: ([\d.]+) \(L1: ([\d.]+), MR: ([\d.]+)\) "
                r"\| MixInterp: ([\d.]+), MixReal: ([\d.]+) \| Rate: ([\d.]+), Gap: ([-\d.]+)",
                line,
            )
            if val_match:
                return {
                    "val_loss": float(val_match.group(1)),
                    "val_recon": float(val_match.group(2)),
                    "val_l1": float(val_match.group(3)),
                    "val_mrstft": float(val_match.group(4)),
                    "val_mix_rate": float(val_match.group(7)),
                }

    return None


def save_progress(output_dir: str, progress: dict):
    """Save pipeline progress to JSON."""
    path = os.path.join(output_dir, "progress.json")
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Stage 1: Training
# ---------------------------------------------------------------------------

def run_training(variant: dict, phase1_checkpoint: str, force: bool = False) -> bool:
    """Train a single variant. Returns True if training completed/skipped successfully."""
    lam = variant["lambda_mix"]
    ckpt_dir = variant["checkpoint_dir"]
    config = variant["config"]

    state = detect_training_state(ckpt_dir)

    if state == "complete" and not force:
        print(f"  [SKIP] lambda={lam}: training already complete")
        return True

    # Determine resume checkpoint
    if state == "partial":
        resume_path = find_latest_checkpoint(ckpt_dir)
        latest_epoch = max(find_checkpoint_files(ckpt_dir))
        print(f"  [RESUME] lambda={lam}: resuming from epoch {latest_epoch}")
    else:
        resume_path = phase1_checkpoint
        print(f"  [START] lambda={lam}: starting from Phase 1 checkpoint")

    cmd = [
        sys.executable, "-m", "training.train",
        "--config", config,
        "--resume", resume_path,
    ]

    print(f"  Command: {' '.join(cmd)}")
    t0 = time.time()

    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"  [DONE] lambda={lam}: training completed in {fmt_time(elapsed)}")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] lambda={lam}: training failed after {fmt_time(elapsed)}")
        print(f"         Return code: {e.returncode}")
        return False


# ---------------------------------------------------------------------------
# Stage 2: Checkpoint Selection
# ---------------------------------------------------------------------------

def run_checkpoint_selection(
    variant: dict, output_dir: str, num_samples: int, force: bool = False
) -> dict | None:
    """Run checkpoint selection for a variant. Returns {selected_epoch, score} or None."""
    lam = variant["lambda_mix"]
    ckpt_dir = variant["checkpoint_dir"]
    output_path = os.path.join(output_dir, f"checkpoint_selection_{lambda_key(lam)}.json")

    # Check for existing results
    if not force and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                data = json.load(f)
            if "recommended_epoch" in data:
                epoch = data["recommended_epoch"]
                score = data["recommended_score"]
                print(f"  [SKIP] lambda={lam}: already selected epoch {epoch} (score={score:.4f})")
                return {"selected_epoch": epoch, "selection_score": score}
        except (json.JSONDecodeError, KeyError):
            pass

    # Reuse existing results for lambda=0.01
    if lam == 0.01 and not force:
        existing = EXISTING_RESULTS["checkpoint_selection"]
        if os.path.exists(existing):
            print(f"  [REUSE] lambda={lam}: copying existing results from {existing}")
            shutil.copy2(existing, output_path)
            with open(output_path) as f:
                data = json.load(f)
            epoch = data["recommended_epoch"]
            score = data["recommended_score"]
            print(f"  Selected epoch {epoch} (score={score:.4f})")
            return {"selected_epoch": epoch, "selection_score": score}

    # Check checkpoint directory exists
    if not os.path.isdir(ckpt_dir):
        print(f"  [SKIP] lambda={lam}: no checkpoint directory found")
        return None

    cmd = [
        sys.executable, "-m", "evaluation.select_checkpoint",
        "--checkpoint-dir", ckpt_dir,
        "--output", output_path,
        "--num-samples", str(num_samples),
    ]

    print(f"  [RUN] lambda={lam}: selecting best checkpoint...")
    t0 = time.time()

    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0

        with open(output_path) as f:
            data = json.load(f)

        epoch = data["recommended_epoch"]
        score = data["recommended_score"]
        print(f"  [DONE] lambda={lam}: epoch {epoch}, score={score:.4f} ({fmt_time(elapsed)})")
        return {"selected_epoch": epoch, "selection_score": score}
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] lambda={lam}: checkpoint selection failed ({fmt_time(elapsed)})")
        print(f"         {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 3: Test Evaluation
# ---------------------------------------------------------------------------

def load_selected_epoch(variant: dict, output_dir: str) -> int | None:
    """Load selected epoch from checkpoint selection JSON."""
    lam = variant["lambda_mix"]
    path = os.path.join(output_dir, f"checkpoint_selection_{lambda_key(lam)}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("recommended_epoch")
    except (json.JSONDecodeError, KeyError):
        return None


def run_test_evaluation(
    variant: dict,
    selected_epoch: int,
    output_dir: str,
    num_samples: int,
    force: bool = False,
) -> dict | None:
    """Run test evaluation for a variant. Returns alpha results dict or None."""
    lam = variant["lambda_mix"]
    ckpt_dir = variant["checkpoint_dir"]
    output_path = os.path.join(output_dir, f"test_eval_{lambda_key(lam)}.json")

    # Check for existing results
    if not force and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                data = json.load(f)
            if "phase2" in data:
                rate = data["phase2"].get("0.5", {}).get("MixRate_mean")
                print(f"  [SKIP] lambda={lam}: already evaluated (MixRate@0.5={rate:.4f})")
                return data["phase2"]
        except (json.JSONDecodeError, KeyError):
            pass

    # Reuse existing results for lambda=0.01
    if lam == 0.01 and not force:
        existing = EXISTING_RESULTS["test_evaluation"]
        if os.path.exists(existing):
            print(f"  [REUSE] lambda={lam}: copying existing results from {existing}")
            shutil.copy2(existing, output_path)
            with open(output_path) as f:
                data = json.load(f)
            rate = data["phase2"]["0.5"]["MixRate_mean"]
            print(f"  MixRate@0.5 = {rate:.4f}")
            return data["phase2"]

    # Construct checkpoint path
    checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_{selected_epoch}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  [SKIP] lambda={lam}: checkpoint not found: {checkpoint_path}")
        return None

    cmd = [
        sys.executable, "-m", "evaluation.test_evaluation",
        "--checkpoint", checkpoint_path,
        "--output", output_path,
        "--num-samples", str(num_samples),
    ]

    print(f"  [RUN] lambda={lam}: evaluating epoch {selected_epoch}...")
    t0 = time.time()

    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0

        with open(output_path) as f:
            data = json.load(f)

        rate = data["phase2"]["0.5"]["MixRate_mean"]
        print(f"  [DONE] lambda={lam}: MixRate@0.5={rate:.4f} ({fmt_time(elapsed)})")
        return data["phase2"]
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] lambda={lam}: test evaluation failed ({fmt_time(elapsed)})")
        print(f"         {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 4: Comparison Table
# ---------------------------------------------------------------------------

def collect_all_results(variants: list, output_dir: str) -> dict:
    """Collect results from all stages into a unified dict keyed by lambda_mix."""
    results = {}
    for variant in variants:
        lam = variant["lambda_mix"]
        entry = {"lambda_mix": lam}

        # Checkpoint selection
        sel_path = os.path.join(output_dir, f"checkpoint_selection_{lambda_key(lam)}.json")
        if os.path.exists(sel_path):
            with open(sel_path) as f:
                sel_data = json.load(f)
            entry["selected_epoch"] = sel_data.get("recommended_epoch")
            entry["selection_score"] = sel_data.get("recommended_score")

        # Test evaluation
        eval_path = os.path.join(output_dir, f"test_eval_{lambda_key(lam)}.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                eval_data = json.load(f)
            phase2 = eval_data.get("phase2", {})
            entry["test_results"] = phase2
            alpha_05 = phase2.get("0.5", {})
            entry["mixrate_05"] = alpha_05.get("MixRate_mean")

        # Training log metrics
        epoch = entry.get("selected_epoch")
        if epoch and os.path.isdir(variant["checkpoint_dir"]):
            log_metrics = parse_log_metrics(variant["checkpoint_dir"], epoch)
            if log_metrics:
                entry.update(log_metrics)

        results[lam] = entry

    return results


def find_best_indices(results: dict) -> dict:
    """Find which lambda has the best value for each metric."""
    lambdas = sorted(results.keys())
    bests = {}

    # Lower is better
    for metric in ["val_loss", "val_l1", "val_mrstft", "selection_score"]:
        values = [(lam, results[lam].get(metric)) for lam in lambdas]
        values = [(lam, v) for lam, v in values if v is not None]
        if values:
            bests[metric] = min(values, key=lambda x: x[1])[0]

    # Closest to 1.0 is best
    mr_values = [(lam, results[lam].get("mixrate_05")) for lam in lambdas]
    mr_values = [(lam, v) for lam, v in mr_values if v is not None]
    if mr_values:
        bests["mixrate_05"] = min(mr_values, key=lambda x: abs(x[1] - 1.0))[0]

    return bests


def generate_terminal_table(results: dict):
    """Print a formatted comparison table to terminal."""
    lambdas = sorted(results.keys())
    bests = find_best_indices(results)

    header = f"{'lambda_mix':>12}  {'Epoch':>6}  {'Val Loss':>10}  {'L1':>8}  {'MR-STFT':>8}  {'MixRate@0.5':>12}  {'Sel. Score':>11}"
    print("=" * len(header))
    print("Lambda_mix Ablation Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for lam in lambdas:
        r = results[lam]
        epoch = r.get("selected_epoch", "?")
        val_loss = f"{r['val_loss']:.4f}" if r.get("val_loss") is not None else "---"
        l1 = f"{r['val_l1']:.4f}" if r.get("val_l1") is not None else "---"
        mr = f"{r['val_mrstft']:.4f}" if r.get("val_mrstft") is not None else "---"
        mixrate = f"{r['mixrate_05']:.4f}" if r.get("mixrate_05") is not None else "---"
        score = f"{r['selection_score']:.4f}" if r.get("selection_score") is not None else "---"

        # Mark best values
        markers = []
        if bests.get("val_loss") == lam:
            markers.append("loss")
        if bests.get("mixrate_05") == lam:
            markers.append("mix")
        if bests.get("selection_score") == lam:
            markers.append("score")
        suffix = f"  <- best {', '.join(markers)}" if markers else ""

        print(f"{lam:>12}  {epoch:>6}  {val_loss:>10}  {l1:>8}  {mr:>8}  {mixrate:>12}  {score:>11}{suffix}")

    print("=" * len(header))


def generate_latex_table(results: dict, output_path: str):
    """Generate LaTeX ablation table."""
    lambdas = sorted(results.keys())
    bests = find_best_indices(results)

    def fmt(val, key, lam, fmt_spec=".3f"):
        if val is None:
            return "---"
        s = f"{val:{fmt_spec}}"
        if bests.get(key) == lam:
            return f"\\textbf{{{s}}}"
        return s

    def fmt_mixrate(val, lam):
        if val is None:
            return "---"
        s = f"{val:.3f}"
        if bests.get("mixrate_05") == lam:
            return f"\\textbf{{{s}}}"
        return s

    latex = r"""\begin{table}[t]
\centering
\caption{Effect of mixing loss weight $\lambda_\text{mix}$ on reconstruction quality
and latent mixing linearity. $\bar{\Delta}$ is the mean absolute deviation of MixRate
from 1.0 across all mixing ratios (lower = better linearity).}
\label{tab:lambda_ablation}
\begin{tabular}{lccccc}
\toprule
$\lambda_\text{mix}$ & Val Loss & L1 & MR-STFT & MixRate ($\alpha\!=\!0.5$) & $\bar{\Delta}$ \\
\midrule
"""

    for lam in lambdas:
        r = results[lam]
        row_parts = [
            str(lam),
            fmt(r.get("val_loss"), "val_loss", lam),
            fmt(r.get("val_l1"), "val_l1", lam),
            fmt(r.get("val_mrstft"), "val_mrstft", lam),
            fmt_mixrate(r.get("mixrate_05"), lam),
            fmt(r.get("selection_score"), "selection_score", lam, ".4f"),
        ]
        latex += " & ".join(row_parts) + " \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table to {output_path}")


def generate_ablation_table(variants: list, output_dir: str):
    """Collect all results and generate both terminal and LaTeX tables."""
    results = collect_all_results(variants, output_dir)

    # Check we have at least some results
    has_data = any(r.get("mixrate_05") is not None for r in results.values())
    if not has_data:
        print("  [WARN] No test evaluation results found. Cannot generate table.")
        print("         Run with --stages select evaluate first.")
        return

    # Terminal table
    print()
    generate_terminal_table(results)

    # LaTeX table
    latex_path = os.path.join(output_dir, "lambda_mix_ablation.tex")
    generate_latex_table(results, latex_path)

    # Save summary JSON
    summary_path = os.path.join(output_dir, "ablation_summary.json")
    # Convert float keys to strings for JSON
    json_results = {str(k): v for k, v in results.items()}
    with open(summary_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Saved summary to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full lambda_mix ablation: train -> select -> evaluate -> table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["train", "select", "evaluate", "table", "all"],
        default=["all"],
        help="Which stages to run",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=None,
        help="Specific lambda_mix values to process (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs exist",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation",
        help="Output directory for ablation results",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        default="checkpoints/musdb-phase1-recon/best_model.pth",
        help="Phase 1 checkpoint to resume from",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of test samples for evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    stages = set(args.stages)
    if "all" in stages:
        stages = {"train", "select", "evaluate", "table"}

    # Filter variants
    variants = VARIANTS
    if args.lambdas:
        variants = [v for v in VARIANTS if v["lambda_mix"] in args.lambdas]
        if not variants:
            print(f"No matching variants for lambdas={args.lambdas}")
            print(f"Available: {[v['lambda_mix'] for v in VARIANTS]}")
            return

    print(f"Ablation pipeline: stages={sorted(stages)}")
    print(f"Variants: {[v['lambda_mix'] for v in variants]}")
    print(f"Output: {args.output_dir}")
    print()

    progress = {}
    t_total = time.time()

    # Stage 1: Training
    if "train" in stages:
        print("=" * 60)
        print("STAGE 1: Training")
        print("=" * 60)
        for variant in variants:
            lam = variant["lambda_mix"]
            ok = run_training(variant, args.phase1_checkpoint, force=args.force)
            progress.setdefault(lambda_key(lam), {})["train"] = "complete" if ok else "failed"
            save_progress(args.output_dir, progress)
        print()

    # Stage 2: Checkpoint Selection
    if "select" in stages:
        print("=" * 60)
        print("STAGE 2: Checkpoint Selection")
        print("=" * 60)
        for variant in variants:
            lam = variant["lambda_mix"]
            result = run_checkpoint_selection(
                variant, args.output_dir, args.num_samples, force=args.force
            )
            status = "complete" if result else "failed"
            progress.setdefault(lambda_key(lam), {})["select"] = status
            save_progress(args.output_dir, progress)
        print()

    # Stage 3: Test Evaluation
    if "evaluate" in stages:
        print("=" * 60)
        print("STAGE 3: Test Evaluation")
        print("=" * 60)
        for variant in variants:
            lam = variant["lambda_mix"]
            selected_epoch = load_selected_epoch(variant, args.output_dir)
            if selected_epoch is None:
                print(f"  [SKIP] lambda={lam}: no selected epoch (run --stages select first)")
                progress.setdefault(lambda_key(lam), {})["evaluate"] = "skipped"
            else:
                result = run_test_evaluation(
                    variant, selected_epoch, args.output_dir,
                    args.num_samples, force=args.force,
                )
                status = "complete" if result else "failed"
                progress.setdefault(lambda_key(lam), {})["evaluate"] = status
            save_progress(args.output_dir, progress)
        print()

    # Stage 4: Table
    if "table" in stages:
        print("=" * 60)
        print("STAGE 4: Comparison Table")
        print("=" * 60)
        generate_ablation_table(variants, args.output_dir)
        print()

    elapsed = time.time() - t_total
    print(f"Total elapsed: {fmt_time(elapsed)}")


if __name__ == "__main__":
    main()
