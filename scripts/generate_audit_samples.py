"""
v1.1 audit samples — per-source 1s chunk pairs + full-track reconstructions.

Loads a checkpoint, walks the test split per source, and writes ref/hat .wav
pairs to a clean output directory. The point of this script is **listening
material**, not metric computation — see evaluation/compute_fad.py for that.

Two outputs per source:
  - chunks/<source>/   4 single-second ref/hat pairs spread across the source's
                       track list (deterministic, same picks every run).
  - tracks/<source>/   3 full-track reconstructions (every chunk of the track
                       encoded → decoded → concatenated). Output is at the
                       dataset's native peak-normalized scale.

Note on full-track recons: each 1s chunk is processed independently, so chunk
boundaries can have small phase discontinuities (audible as soft clicks every
1s). This is a known artifact of fixed-window AE inference; for a clean
demo-grade reconstruction you'd want overlap-add. For audit purposes the
discontinuities are tolerable.

Run:
  python -m scripts.generate_audit_samples \
      --config checkpoints/v1.1/config.yaml \
      --checkpoint checkpoints/v1.1/best.pth \
      --out evaluation/audit_v1.1
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from torch.amp import autocast
from tqdm import tqdm

from data.dataset import WaveformDataset
from models.autoencoder import Autoencoder
from training.config import build_model_config, get_device, load_config


def _spread_indices(n: int, k: int) -> List[int]:
    """Pick k indices spread across [0, n). For k=4, n=100 -> [0, 33, 66, 99]."""
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def _safe_stem(key: str, max_len: int = 60) -> str:
    """Make a track key filesystem-safe and trimmed."""
    out = key.replace("/", "_").replace("\\", "_").replace(":", "_")
    return out[:max_len]


@torch.no_grad()
def reconstruct_full_track(
    model: Autoencoder,
    dataset: WaveformDataset,
    file_info: Dict,
    device: torch.device,
    batch_size: int = 8,
) -> tuple:
    """Encode every chunk of one file in order, decode, concatenate.

    Returns (ref_full, hat_full) as float32 numpy arrays at the dataset's SR.
    """
    chunk_idxs = list(range(file_info["start"], file_info["end"]))
    refs, hats = [], []

    for i in range(0, len(chunk_idxs), batch_size):
        batch = []
        for ci in chunk_idxs[i:i + batch_size]:
            item = dataset[ci]
            wav = item["x_wave"]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            batch.append(wav)
        x = torch.stack(batch, dim=0).to(device)
        _, _, x_hat, _ = model(x)
        tgt = model.decoder.target_length
        x_ref = x[:, :, :tgt].float().cpu()
        x_hat_cpu = x_hat.float().cpu()
        for r, h in zip(x_ref.squeeze(1), x_hat_cpu.squeeze(1)):
            refs.append(r.numpy())
            hats.append(h.numpy())

    return (
        np.concatenate(refs).astype(np.float32),
        np.concatenate(hats).astype(np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="checkpoints/v1.1/config.yaml")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/v1.1/best.pth")
    ap.add_argument("--out", type=str, default="evaluation/audit_v1.1")
    ap.add_argument("--chunks-per-source", type=int, default=4,
                    help="Number of 1s ref/hat pairs per source")
    ap.add_argument("--tracks-per-source", type=int, default=3,
                    help="Number of full-track recons per source")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Decode batch size (lower if OOM)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    sample_rate = int(cfg["data"]["sample_rate"])
    chunks_dir = cfg["data"]["chunks_dir"]

    # Build model + load weights
    model = Autoencoder(**build_model_config(cfg)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("global_step", "?")
    print(f"Loaded {args.checkpoint} @ step {step}")

    # Load val dataset
    val_ds = WaveformDataset(chunks_dir, split="test")
    print(f"Val: {len(val_ds):,} chunks across {len(val_ds.files)} files")

    # Group files by source
    by_source: Dict[str, List[Dict]] = {}
    for f in val_ds.files:
        by_source.setdefault(f["source"], []).append(f)

    chunks_root = os.path.join(args.out, "chunks")
    tracks_root = os.path.join(args.out, "tracks")

    for source in sorted(by_source.keys()):
        files = by_source[source]
        print(f"\n=== {source}: {len(files)} files ===")

        # 1-second chunk pairs
        chunk_dir = os.path.join(chunks_root, source)
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_picks = _spread_indices(len(files), args.chunks_per_source)
        for pi, fi in enumerate(chunk_picks):
            f = files[fi]
            item = val_ds[f["start"]]  # first chunk of this file
            x = item["x_wave"]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x = x.unsqueeze(0).to(device)  # [1, 1, L]
            with torch.no_grad():
                _, _, x_hat, _ = model(x)
            tgt = model.decoder.target_length
            ref = x[:, :, :tgt].float().cpu().squeeze().numpy()
            hat = x_hat.float().cpu().squeeze().numpy()
            stem = _safe_stem(f["key"])
            sf.write(os.path.join(chunk_dir, f"{pi:02d}_{stem}_ref.wav"), ref, sample_rate)
            sf.write(os.path.join(chunk_dir, f"{pi:02d}_{stem}_hat.wav"), hat, sample_rate)
        print(f"  chunks: {len(chunk_picks)} pairs -> {chunk_dir}")

        # Full-track reconstructions
        track_dir = os.path.join(tracks_root, source)
        os.makedirs(track_dir, exist_ok=True)
        track_picks = _spread_indices(len(files), args.tracks_per_source)
        for pi, fi in enumerate(tqdm(track_picks, desc=f"  tracks/{source}")):
            f = files[fi]
            ref_full, hat_full = reconstruct_full_track(
                model, val_ds, f, device, batch_size=args.batch_size
            )
            duration = len(ref_full) / sample_rate
            stem = _safe_stem(f["key"])
            sf.write(os.path.join(track_dir, f"{pi:02d}_{stem}_ref.wav"),
                     ref_full, sample_rate)
            sf.write(os.path.join(track_dir, f"{pi:02d}_{stem}_hat.wav"),
                     hat_full, sample_rate)

    print(f"\nAudit samples written to: {args.out}")
    print(f"  chunks: {chunks_root}")
    print(f"  tracks: {tracks_root}")


if __name__ == "__main__":
    main()
