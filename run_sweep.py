"""
Run Phase 2 λ_mix sweep: 6 values × 20 epochs each.
All resume from the Phase 1 2D checkpoint.

Usage:
    python run_sweep.py
    python run_sweep.py --start-from 0.05   # skip values below 0.05
"""

import subprocess
import sys
import argparse
from pathlib import Path

PYTHON = sys.executable
CHECKPOINT = "checkpoints/musdb-phase1-2d/best_model.pth"
SWEEP_DIR = "configs/experiments/sweep"

LAMBDA_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def run_training(lambda_mix: float) -> int:
    config = f"{SWEEP_DIR}/musdb_phase2_2d_mix{lambda_mix}.yaml"
    if not Path(config).exists():
        print(f"Config not found: {config}")
        return 1

    cmd = [
        PYTHON, "-m", "training.train",
        "--config", config,
        "--resume", CHECKPOINT,
    ]
    print(f"\n{'='*60}")
    print(f"  Starting sweep: λ_mix = {lambda_mix}")
    print(f"  Config: {config}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=float, default=0.0,
                        help="Skip lambda values below this (for resuming)")
    args = parser.parse_args()

    values = [v for v in LAMBDA_VALUES if v >= args.start_from]
    print(f"Sweep plan: λ_mix ∈ {values}")
    print(f"Checkpoint: {CHECKPOINT}")

    for i, lam in enumerate(values):
        print(f"\n[{i+1}/{len(values)}] Running λ_mix = {lam}...")
        rc = run_training(lam)
        if rc != 0:
            print(f"Training failed for λ_mix = {lam} (exit code {rc})")
            print("Stopping sweep. Fix the issue and rerun with --start-from")
            sys.exit(rc)
        print(f"Completed λ_mix = {lam}")

    print(f"\nSweep complete! All {len(values)} runs finished.")
    print("Next: run analysis to find Pareto-optimal λ_mix")


if __name__ == "__main__":
    main()
