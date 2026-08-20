"""Run compute_mixing_metrics against an M2L checkpoint via the adapter.

Reuses ``_process_batch`` and ``_tally`` from
``evaluation.compute_mixing_metrics`` verbatim — the adapter exposes the same
attributes the metric code reads on the v2 Autoencoder.

Usage:
  python -m evaluation.m2l_run_mixing \
      --m2l-checkpoint $MUSICGEN_M2L_PUBLISHED \
      --musicgen-config configs/experiments/v2/v2.0_continued.yaml \
      --out evaluation/v2_metrics/m2l_phase0_mixing.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import torch
from tqdm import tqdm

from data.dataset import build_dataloaders
from training.config import get_device, load_config

from evaluation.compute_mixing_metrics import _process_batch, _tally
from evaluation.m2l_adapter import M2LAutoencoderAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2l-checkpoint", type=str, default=None,
                    help="Path to a music2latent .pt checkpoint. If omitted, the "
                         "M2L EncoderDecoder default (auto-downloaded music2latent.pt) is used.")
    ap.add_argument("--musicgen-config", type=str,
                    default="configs/experiments/v2/v2.0_continued.yaml",
                    help="v2 config used only for chunks_dir / num_workers — the model "
                         "comes from --m2l-checkpoint, not from this config.")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Cap eval batches (smoke test)")
    ap.add_argument("--per-source", type=int, default=None,
                    help="Stratified subsample: this many chunks PER source, "
                         "balanced. Preferred over --max-batches.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.musicgen_config)
    device = get_device()

    print(f"loading M2L checkpoint: {args.m2l_checkpoint or '<default>'}")
    model = M2LAutoencoderAdapter(
        m2l_checkpoint_path=args.m2l_checkpoint,
        device=device,
    ).to(device)
    model.eval()

    # Subsampling: --per-source N gives a balanced N/source draw (preferred);
    # --max-batches gives a proportional shuffled draw (fma-dominated, legacy).
    _, val_loader, _, val_ds, _ = build_dataloaders(
        chunks_dir=cfg["data"]["chunks_dir"],
        batch_size=args.batch_size,
        num_workers=int(cfg.get("train", {}).get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
        val_per_source=args.per_source,
        val_shuffle=(args.per_source is None and args.max_batches is not None),
        val_seed=args.seed,
    )
    print(f"val: {len(val_ds):,} chunks across {len(val_ds.files)} files")

    aggregates: Dict[str, List[tuple]] = {
        "sdr_rec": [], "sdr_lin": [], "sdr_lin_gt": [], "l_lat": [], "mix_rate": [],
    }

    n_seen = 0
    for bi, batch in enumerate(tqdm(val_loader, desc="m2l-mixing")):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        x_wave = batch["x_wave"].to(device, non_blocking=True)
        sources = batch.get("source", None)
        if sources is None:
            sources = ["unknown"] * x_wave.size(0)
        elif not isinstance(sources, list):
            sources = list(sources)

        out = _process_batch(model, x_wave, sources, alpha=args.alpha)
        for k in aggregates:
            aggregates[k].extend(out.get(k, []))
        n_seen += x_wave.size(0)

    summary = {k: _tally(v) for k, v in aggregates.items()}

    from collections import Counter as _Counter
    src_counts = _Counter(s for s, _ in aggregates["sdr_lin"])
    print(f"per-source samples: {dict(src_counts)}  (n_seen={n_seen})")

    print("\n=== M2L mixing metrics ===")
    for metric in ("sdr_rec", "sdr_lin", "sdr_lin_gt", "l_lat", "mix_rate"):
        line = f"  {metric:8s}"
        for src in sorted(summary[metric].keys()):
            line += f"  {src}={summary[metric][src]:+.4f}"
        print(line)

    out_dict = {
        "checkpoint": args.m2l_checkpoint or "<music2latent default>",
        "model": "music2latent",
        "step": -1,
        "alpha": args.alpha,
        "n_samples_seen": n_seen,
        "metrics": summary,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
