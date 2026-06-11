"""Run compute_fad against an M2L checkpoint via the adapter.

Reuses ``_reconstruct_full_track``, ``_to_clap_clips``, ``_frechet_distance``
from ``evaluation.compute_fad`` verbatim; only the model construction is
different.

Usage:
  python -m evaluation.m2l_run_fad \
      --m2l-checkpoint d:/projects/music2latent/music2latent/models/music2latent.pt \
      --musicgen-config configs/experiments/v2/v2.0_continued.yaml \
      --out evaluation/v2_metrics/m2l_phase0_fad.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from data.dataset import WaveformDataset
from training.config import get_device, load_config

from evaluation.compute_fad import (
    CLAP_SR, CLAP_CLIP_SECONDS,
    _reconstruct_full_track, _to_clap_clips, _frechet_distance,
)
from evaluation.m2l_adapter import M2LAutoencoderAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2l-checkpoint", type=str, default=None,
                    help="Path to a music2latent .pt checkpoint. If omitted, the "
                         "M2L EncoderDecoder default (auto-downloaded music2latent.pt) is used.")
    ap.add_argument("--musicgen-config", type=str,
                    default="configs/experiments/v2/v2.0_continued.yaml",
                    help="v2 config used only for chunks_dir / sample_rate — model "
                         "comes from --m2l-checkpoint, not from this config.")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Decode batch size for full-track reconstruction.")
    ap.add_argument("--max-tracks-per-source", type=int, default=None,
                    help="Cap tracks per source (smoke test).")
    ap.add_argument("--clap-ckpt", type=str,
                    default="checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
    args = ap.parse_args()

    cfg = load_config(args.musicgen_config)
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

    print(f"loading CLAP from {args.clap_ckpt}...")
    import laion_clap
    clap = laion_clap.CLAP_Module(
        enable_fusion=False, amodel="HTSAT-base"
    ).to(device)
    clap.load_ckpt(ckpt=args.clap_ckpt)
    clap.eval()

    print(f"loading M2L checkpoint: {args.m2l_checkpoint or '<default>'}")
    model = M2LAutoencoderAdapter(
        m2l_checkpoint_path=args.m2l_checkpoint,
        device=device,
    ).to(device)
    model.eval()

    val_ds = WaveformDataset(cfg["data"]["chunks_dir"], split="test")
    by_source: Dict[str, List[Dict]] = {}
    for f in val_ds.files:
        by_source.setdefault(f["source"], []).append(f)

    @torch.no_grad()
    def embed_one(wav48k: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(wav48k).unsqueeze(0).to(device)
        emb = clap.get_audio_embedding_from_data(x=x, use_tensor=True)
        return emb.float().cpu().numpy()

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

    real_by_src["all"] = np.concatenate(list(real_by_src.values()), axis=0)
    fake_by_src["all"] = np.concatenate(list(fake_by_src.values()), axis=0)

    fad_scores = {
        f"fad/{src}": _frechet_distance(real_by_src[src], fake_by_src[src])
        for src in real_by_src
    }

    print("\n=== M2L FAD scores ===")
    for k in sorted(fad_scores.keys()):
        print(f"  {k}: {fad_scores[k]:.4f}")

    out_dict = {
        "checkpoint": args.m2l_checkpoint or "<music2latent default>",
        "model": "music2latent",
        "step": -1,
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
