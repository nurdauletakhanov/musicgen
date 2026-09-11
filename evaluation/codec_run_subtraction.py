"""Latent stem subtraction, raw and origin-recentered, on third-party
autoencoders (DAC, EnCodec, Lin-CAE and its retrained Music2Latent baseline).

Reuses ``run_eval`` / ``print_summary`` from ``evaluation.compute_subtraction``
unchanged, so units, metrics, comparators and the per-chunk record format are
identical to the GAN and Music2Latent evaluations and the same paired
statistics scripts apply.

Usage:
  python -m evaluation.codec_run_subtraction --model dac44k \
      --out evaluation/v2_metrics/codecs/dac44k_subtraction.json \
      --per-chunk-out evaluation/v2_metrics/codecs/per_chunk/dac44k_per_chunk.json
  # add --origin-correct for f(mix) - f(stem) + f(0)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from evaluation.codec_adapters import CHUNK_LEN, MODEL_IDS, CodecAdapter
from evaluation.compute_subtraction import print_summary, run_eval
from training.config import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--musdb-dir", default="dataset/musdb18/test")
    ap.add_argument("--chunks-per-track", type=int, default=30)
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="override (cpu / cuda / mps)")
    ap.add_argument("--origin-correct", action="store_true",
                    help="decode f(mix) - f(stem) + f(0) instead of f(mix) - f(stem)")
    ap.add_argument("--per-chunk-out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    model = CodecAdapter(args.model, device)
    info = model.describe()
    print("loaded", json.dumps(info))

    origin = None
    origin_stats = None
    with torch.no_grad():
        torch.manual_seed(args.seed)
        z0 = model.encoder(torch.zeros(1, 1, CHUNK_LEN, device=device))
        origin_stats = {"f0_norm": float(z0.norm()), "f0_rms": float(z0.pow(2).mean().sqrt()),
                        "latent_shape": list(z0.shape[1:])}
        print(f"f(0): norm {origin_stats['f0_norm']:.4f}, per-element RMS {origin_stats['f0_rms']:.4f}, "
              f"shape {origin_stats['latent_shape']}")
        if args.origin_correct:
            origin = z0

    per_chunk = [] if args.per_chunk_out else None
    summary, n_seen, skipped = run_eval(
        model=model, musdb_dir=args.musdb_dir, chunks_per_track=args.chunks_per_track,
        batch_size=args.batch_size, seed=args.seed, device=device,
        max_tracks=args.max_tracks, desc=f"subtraction/{args.model}",
        per_chunk=per_chunk, origin=origin)
    print_summary(summary, n_seen, skipped)

    out = {
        "checkpoint": info["hf_id"], "model": info, "step": -1,
        "config": {"musdb_dir": args.musdb_dir, "chunks_per_track": args.chunks_per_track,
                   "n_chunks_seen": n_seen, "seed": args.seed},
        "skipped_tracks": skipped,
        "origin_corrected": bool(args.origin_correct),
        "origin": origin_stats,
        "subtraction": summary,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)
    if per_chunk is not None:
        os.makedirs(os.path.dirname(args.per_chunk_out) or ".", exist_ok=True)
        with open(args.per_chunk_out, "w") as f:
            json.dump({"checkpoint": info["hf_id"], "seed": args.seed,
                       "chunks_per_track": args.chunks_per_track, "records": per_chunk}, f)
        print(f"wrote {args.per_chunk_out} ({len(per_chunk)} measurements)")


if __name__ == "__main__":
    main()
