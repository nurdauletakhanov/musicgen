"""
Per-source FAD on the v1.1 test set, using LAION-CLAP embeddings.

For every test track:
  1. Reconstruct the full track (encode every 1s chunk → decode → concat).
  2. Resample reference and reconstruction to 48 kHz (CLAP's expected SR).
  3. Slice each into non-overlapping 10 s windows (CLAP's natural window).
     Tracks shorter than 10 s fall back to a single padded center clip.
  4. Embed every clip via LAION-CLAP HTSAT-base.

Multiple clips per track keeps the Frechet covariance well-conditioned at
512 dims — single-clip-per-track left us at ~50 embeddings/source, which
is unstable for FAD. Clips from the same track aren't independent, but
the covariance estimate is materially better than with 50 samples.

Then compute Frechet distance between {real} and {fake} embedding sets,
both per-source and overall. One number per source = one row of the audit table.

Why CLAP and not VGGish-FAD: the standard `frechet_audio_distance` package
imports numba internals that are blocked by Windows AppControl on this
machine. CLAP is torch-native, music-trained, and gives FAD numbers well
correlated with perceived quality on music.

Prerequisite: download the music-finetuned HTSAT-base CLAP checkpoint
(~2 GB) once, place it at checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt
(or pass --clap-ckpt). Source:
  https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt

Usage:
  python -m evaluation.compute_fad \
      --config checkpoints/v1.1/config.yaml \
      --checkpoint checkpoints/v1.1/best.pth \
      --out evaluation/audit_v1.1/fad.json

Smoke test (3 tracks per source, ~2 min):
  python -m evaluation.compute_fad --max-tracks-per-source 3 \
      --out evaluation/audit_v1.1/fad_smoke.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.signal import resample_poly
from tqdm import tqdm

from data.dataset import WaveformDataset
from models.autoencoder import Autoencoder
from training.config import build_model_config, get_device, load_config


CLAP_SR = 48000
CLAP_CLIP_SECONDS = 10.0  # CLAP processes audio in ~10s windows internally


# ---------------------------------------------------------------------------
# Helpers

@torch.no_grad()
def _reconstruct_full_track(
    model: Autoencoder,
    dataset: WaveformDataset,
    file_info: Dict,
    device: torch.device,
    batch_size: int = 8,
) -> tuple:
    """Encode → decode every chunk of one file in order, concatenate."""
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


def _to_clap_clips(audio: np.ndarray, src_sr: int) -> List[np.ndarray]:
    """Resample to 48 kHz, then split into non-overlapping CLAP_CLIP_SECONDS windows.

    Returns a list of float32 arrays, each exactly CLAP_CLIP_SECONDS * 48000 samples.
    Trailing partial windows are dropped. Tracks shorter than one window are
    center-padded to a single clip so they still contribute one embedding.
    """
    if src_sr != CLAP_SR:
        from math import gcd
        g = gcd(int(src_sr), CLAP_SR)
        audio = resample_poly(audio, CLAP_SR // g, src_sr // g)
    audio = audio.astype(np.float32)

    win = int(CLAP_CLIP_SECONDS * CLAP_SR)
    n = len(audio)
    if n < win:
        pad = np.zeros(win, dtype=np.float32)
        start = (win - n) // 2
        pad[start:start + n] = audio
        return [pad]
    n_clips = n // win
    return [audio[i * win:(i + 1) * win] for i in range(n_clips)]


def _frechet_distance(real: np.ndarray, fake: np.ndarray, eps: float = 1e-6) -> float:
    """Frechet distance between two multivariate Gaussians fit to each set."""
    mu_r, mu_f = real.mean(0), fake.mean(0)
    sig_r = np.cov(real, rowvar=False)
    sig_f = np.cov(fake, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = sqrtm(sig_r.dot(sig_f), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        covmean = sqrtm(
            (sig_r + eps * np.eye(sig_r.shape[0])).dot(
                sig_f + eps * np.eye(sig_f.shape[0])
            )
        )
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r) + np.trace(sig_f) - 2 * np.trace(covmean))


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="checkpoints/v1.1/config.yaml")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/v1.1/best.pth")
    ap.add_argument("--out", type=str, default="evaluation/audit_v1.1/fad.json")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Decode batch size (lower if OOM)")
    ap.add_argument("--max-tracks-per-source", type=int, default=None,
                    help="Cap tracks per source (smoke test)")
    ap.add_argument("--clap-ckpt", type=str,
                    default="checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt",
                    help="Music-finetuned HTSAT-base CLAP checkpoint. Download from "
                         "https://huggingface.co/lukewys/laion_clap")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    src_sr = int(cfg["data"]["sample_rate"])

    if not os.path.isfile(args.clap_ckpt):
        raise FileNotFoundError(
            f"CLAP checkpoint not found: {args.clap_ckpt}\n"
            f"Download music_audioset_epoch_15_esc_90.14.pt (~2 GB) from\n"
            f"  https://huggingface.co/lukewys/laion_clap/resolve/main/"
            f"music_audioset_epoch_15_esc_90.14.pt\n"
            f"and place it at the path above (or pass --clap-ckpt)."
        )

    # Load CLAP first — heaviest setup, fail fast if it doesn't initialize
    print(f"loading CLAP (HTSAT-base, music-finetuned) from {args.clap_ckpt}...")
    import laion_clap
    clap = laion_clap.CLAP_Module(
        enable_fusion=False, amodel="HTSAT-base"
    ).to(device)
    clap.load_ckpt(ckpt=args.clap_ckpt)
    clap.eval()

    # Load AE
    model = Autoencoder(**build_model_config(cfg)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("global_step", "?")
    print(f"loaded checkpoint @ step {step}")

    val_ds = WaveformDataset(cfg["data"]["chunks_dir"], split="test")
    by_source: Dict[str, List[Dict]] = {}
    for f in val_ds.files:
        by_source.setdefault(f["source"], []).append(f)

    @torch.no_grad()
    def embed_one(wav48k: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(wav48k).unsqueeze(0).to(device)  # [1, samples]
        emb = clap.get_audio_embedding_from_data(x=x, use_tensor=True)
        return emb.float().cpu().numpy()  # [1, 512]

    real_by_src: Dict[str, np.ndarray] = {}
    fake_by_src: Dict[str, np.ndarray] = {}

    for source, files in by_source.items():
        if args.max_tracks_per_source:
            files = files[:args.max_tracks_per_source]
        real_list, fake_list = [], []
        for f in tqdm(files, desc=f"fad/{source}"):
            ref_full, hat_full = _reconstruct_full_track(
                model, val_ds, f, device, batch_size=args.batch_size
            )
            ref_clips = _to_clap_clips(ref_full, src_sr)
            hat_clips = _to_clap_clips(hat_full, src_sr)
            for clip in ref_clips:
                real_list.append(embed_one(clip))
            for clip in hat_clips:
                fake_list.append(embed_one(clip))
        real_by_src[source] = np.concatenate(real_list, axis=0)
        fake_by_src[source] = np.concatenate(fake_list, axis=0)
        print(f"  {source}: {real_by_src[source].shape[0]} embeddings "
              f"from {len(files)} tracks")

    # Aggregate "all" pool
    real_by_src["all"] = np.concatenate(list(real_by_src.values()), axis=0)
    fake_by_src["all"] = np.concatenate(list(fake_by_src.values()), axis=0)

    fad_scores = {
        f"fad/{src}": _frechet_distance(real_by_src[src], fake_by_src[src])
        for src in real_by_src
    }

    print("\n=== FAD scores ===")
    for k in sorted(fad_scores.keys()):
        print(f"  {k}: {fad_scores[k]:.4f}")

    out_dict = {
        "checkpoint": args.checkpoint,
        "step": int(ckpt.get("global_step", -1)) if isinstance(ckpt.get("global_step"), int) else -1,
        "n_embeddings_per_source": {
            s: int(real_by_src[s].shape[0]) for s in real_by_src
        },
        "config": {
            "sample_rate": src_sr,
            "clap_sr": CLAP_SR,
            "clip_seconds": CLAP_CLIP_SECONDS,
        },
        **fad_scores,
    }
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
