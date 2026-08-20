"""Run compute_subtraction.py against an M2L checkpoint via the adapter.

Reuses ``run_eval`` and ``print_summary`` from
``evaluation.compute_subtraction`` verbatim — the M2L adapter exposes
``encoder`` / ``decoder`` matching the v2 Autoencoder surface that the eval
expects. The per-decode reseeding inside ``_process_track`` makes M2L's
consistency-model decode deterministic across the (subtraction, ceiling) pair.

Usage:
  python -m evaluation.m2l_run_subtraction \
      --m2l-checkpoint $MUSICGEN_M2L_CHECKPOINT \
      --out evaluation/v2_metrics/m2l_phase2_subtraction.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from training.config import get_device

from evaluation.compute_subtraction import run_eval, print_summary
from evaluation.m2l_adapter import M2LAutoencoderAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2l-checkpoint", type=str, default=None,
                    help="Path to a music2latent .pt checkpoint. Default = vanilla "
                         "music2latent.pt auto-downloaded by the M2L library.")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--musdb-dir", type=str, default="dataset/musdb18/test")
    ap.add_argument("--chunks-per-track", type=int, default=30)
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Smaller default than v2 — M2L decode is slower.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    print(f"loading M2L checkpoint: {args.m2l_checkpoint or '<default>'}")
    model = M2LAutoencoderAdapter(
        m2l_checkpoint_path=args.m2l_checkpoint,
        device=device,
    ).to(device)
    model.eval()

    summary, n_seen, skipped = run_eval(
        model=model,
        musdb_dir=args.musdb_dir,
        chunks_per_track=args.chunks_per_track,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        max_tracks=args.max_tracks,
        desc="m2l-subtraction",
    )

    print_summary(summary, n_seen, skipped)

    out_dict = {
        "checkpoint": args.m2l_checkpoint or "<music2latent default>",
        "model": "music2latent",
        "step": -1,
        "config": {
            "musdb_dir": args.musdb_dir,
            "chunks_per_track": args.chunks_per_track,
            "n_chunks_seen": n_seen,
            "seed": args.seed,
        },
        "skipped_tracks": skipped,
        "subtraction": summary,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
