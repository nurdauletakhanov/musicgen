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
    ap.add_argument("--origin-correct", action="store_true",
                    help="Decode f(mix) - f(stem) + f(0) (origin correction).")
    ap.add_argument("--per-chunk-out", type=str, default=None,
                    help="Also dump every measurement (JSON) for paired statistics.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    print(f"loading M2L checkpoint: {args.m2l_checkpoint or '<default>'}")
    model = M2LAutoencoderAdapter(
        m2l_checkpoint_path=args.m2l_checkpoint,
        device=device,
    ).to(device)
    model.eval()

    origin = None
    if args.origin_correct:
        from evaluation.compute_subtraction import CHUNK_LEN
        with torch.no_grad():
            torch.manual_seed(args.seed)
            origin = model.encoder(torch.zeros(1, 1, CHUNK_LEN, device=device))
        print(f"origin correction: f(0) latent norm {origin.norm().item():.4f}")
    per_chunk = [] if args.per_chunk_out else None
    summary, n_seen, skipped = run_eval(
        model=model,
        origin=origin, per_chunk=per_chunk,
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
    if per_chunk is not None:
        import os as _os
        _os.makedirs(_os.path.dirname(args.per_chunk_out) or ".", exist_ok=True)
        with open(args.per_chunk_out, "w") as f:
            json.dump({"checkpoint": args.m2l_checkpoint, "seed": args.seed,
                       "origin_corrected": bool(args.origin_correct),
                       "records": per_chunk}, f)
        print(f"wrote {args.per_chunk_out} ({len(per_chunk)} measurements)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
