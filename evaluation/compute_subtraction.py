"""
Per-stem latent-subtraction eval on MUSDB18-HQ test.

For every test track, picks N non-silent 1-second windows aligned across the
5 stems, then for each of {drums, bass, vocals, other}:

  z_full = f(x_mixture)
  z_stem = f(x_stem_to_remove)
  x_hat  = g(z_full - z_stem)              # latent subtraction
  target = x_mixture - x_stem              # ground truth (sum of remaining stems)
  x_ceil = g(f(target))                    # encode-decode ceiling

Reports per-stem mean SI-SDR for both x_hat and x_ceil, plus the gap (the
"linearity tax"). MUSDB-only by construction — no fma/maestro source split.

Model is loaded as a v2 Autoencoder by default. The internal ``run_eval``
function is model-agnostic; ``m2l_run_subtraction.py`` reuses it with the M2L
adapter.

Usage:
  python -m evaluation.compute_subtraction \
      --config configs/experiments/v3/v3.1_decmix_disc_d64.yaml \
      --checkpoint checkpoints/v3.1-decmix-disc-d64/best.pth \
      --out evaluation/v2_metrics/v3.1-decmix-disc-d64_subtraction.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from models.autoencoder import Autoencoder
from training.config import build_model_config, get_device, load_config


STEMS = ["drums", "bass", "vocals", "other"]
SR = 44100
CHUNK_LEN = SR  # 1 s


# ---------------------------------------------------------------------------
# Per-sample SI-SDR (vectorized over batch)

def _si_sdr_batch(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR (dB) per sample. x_hat, x: [B, T]. Returns [B]."""
    x_hat = x_hat.float()
    x = x.float()
    x = x - x.mean(dim=1, keepdim=True)
    x_hat = x_hat - x_hat.mean(dim=1, keepdim=True)
    alpha = (x_hat * x).sum(dim=1, keepdim=True) / (x.pow(2).sum(dim=1, keepdim=True) + eps)
    s_target = alpha * x
    e_noise = x_hat - s_target
    return 10.0 * torch.log10(
        (s_target.pow(2).sum(dim=1) + eps) / (e_noise.pow(2).sum(dim=1) + eps)
    )


# ---------------------------------------------------------------------------
# MUSDB IO

def _load_track_stems(track_dir: Path) -> Dict[str, np.ndarray]:
    """Load 5 stems for a MUSDB track as mono float32 @ 44.1 kHz."""
    out: Dict[str, np.ndarray] = {}
    for stem in ["mixture"] + STEMS:
        wav, sr = sf.read(str(track_dir / f"{stem}.wav"),
                          dtype="float32", always_2d=True)
        if sr != SR:
            raise RuntimeError(f"unexpected sr={sr} in {track_dir / stem}")
        out[stem] = wav.mean(axis=1)  # stereo -> mono
    return out


def _stem_sanity(stems: Dict[str, np.ndarray]) -> float:
    """SI-SDR of summed stems vs mixture (numpy, scalar). High = identity holds."""
    summed = stems["drums"] + stems["bass"] + stems["vocals"] + stems["other"]
    n = min(len(summed), len(stems["mixture"]))
    a = summed[:n].astype(np.float64)
    b = stems["mixture"][:n].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    alpha = (a * b).sum() / max((b * b).sum(), 1e-8)
    target = alpha * b
    noise = a - target
    denom = (noise ** 2).sum()
    if denom <= 0:
        return 200.0
    return float(10.0 * np.log10(max((target ** 2).sum(), 1e-12) / denom))


def _pick_chunks(stems: Dict[str, np.ndarray], n_chunks: int,
                 rng: random.Random, rms_thresh: float = 1e-3) -> List[int]:
    """Random non-silent chunk start indices, aligned across all stems."""
    n = min(len(s) for s in stems.values())
    if n < CHUNK_LEN:
        return []
    starts: List[int] = []
    max_tries = n_chunks * 50
    tries = 0
    while len(starts) < n_chunks and tries < max_tries:
        s = rng.randrange(0, n - CHUNK_LEN)
        seg = stems["mixture"][s:s + CHUNK_LEN]
        if np.sqrt(np.mean(seg ** 2)) >= rms_thresh:
            starts.append(s)
        tries += 1
    return starts


# ---------------------------------------------------------------------------
# Core eval — model-agnostic

@torch.no_grad()
def _process_track(model, stems: Dict[str, np.ndarray], chunk_starts: List[int],
                   device: torch.device, batch_size: int,
                   seed: int) -> Dict[str, Dict[str, List[float]]]:
    """For one track, run the subtraction + ceiling ops for every chunk and stem.

    Returns {stem: {"sub": [...], "ceil": [...], "dd": [...]}} (SI-SDR per chunk).

    "dd" is the decode-vs-decode variant SI-SDR(x_sub, x_ceil): both args are
    decodes sharing the same init noise, so for consistency-model decoders
    (M2L) the phase nuisance cancels and the number isolates the latent-
    arithmetic error from the model's phase incoherence vs raw waveforms.

    Re-seeds the global RNG before each encoder / decoder call so that M2L's
    consistency-model decoder uses the same starting noise for the two decodes
    being compared (sub vs ceil). For v2 models the encoder/decoder are
    deterministic, so the reseeding is a no-op.
    """
    n = len(chunk_starts)
    mix_arr = np.stack([stems["mixture"][s:s + CHUNK_LEN] for s in chunk_starts])
    stem_arr = {st: np.stack([stems[st][s:s + CHUNK_LEN] for s in chunk_starts])
                for st in STEMS}
    x_mix_all = torch.from_numpy(mix_arr).to(device)              # [n, T]
    x_stem_all = {st: torch.from_numpy(stem_arr[st]).to(device)
                  for st in STEMS}

    out = {st: {"sub": [], "ceil": [], "dd": []} for st in STEMS}

    for st in STEMS:
        for i in range(0, n, batch_size):
            xm = x_mix_all[i:i + batch_size]
            xs = x_stem_all[st][i:i + batch_size]
            tg = xm - xs                                          # ground truth

            torch.manual_seed(seed)
            z_full = model.encoder(xm.unsqueeze(1))
            torch.manual_seed(seed)
            z_stem = model.encoder(xs.unsqueeze(1))
            torch.manual_seed(seed)
            x_sub, _ = model.decoder(z_full - z_stem)
            x_sub = x_sub.squeeze(1)

            torch.manual_seed(seed)
            z_tg = model.encoder(tg.unsqueeze(1))
            torch.manual_seed(seed)
            x_ceil, _ = model.decoder(z_tg)
            x_ceil = x_ceil.squeeze(1)

            # Crop to shortest length (M2L decoder is 42496; v2 is 44100)
            L = min(x_sub.size(1), tg.size(1), x_ceil.size(1))
            sub_sdr = _si_sdr_batch(x_sub[:, :L], tg[:, :L]).cpu().tolist()
            ceil_sdr = _si_sdr_batch(x_ceil[:, :L], tg[:, :L]).cpu().tolist()
            dd_sdr = _si_sdr_batch(x_sub[:, :L], x_ceil[:, :L]).cpu().tolist()
            out[st]["sub"].extend(sub_sdr)
            out[st]["ceil"].extend(ceil_sdr)
            out[st]["dd"].extend(dd_sdr)

    return out


def run_eval(model, musdb_dir: str, chunks_per_track: int, batch_size: int,
             seed: int, device: torch.device, max_tracks: int = None,
             sanity_thresh: float = 20.0,
             desc: str = "subtraction") -> tuple:
    """Iterate MUSDB tracks, aggregate per-stem SI-SDR.

    Returns (summary_dict, n_chunks_seen, skipped_tracks).
    summary_dict[stem] = {"sdr_sub", "sdr_ceil", "sdr_dd", "gap", "n"}; plus "all" entry.
    """
    rng = random.Random(seed)
    root = Path(musdb_dir)
    if not root.exists():
        raise SystemExit(f"musdb dir not found: {root}")

    tracks = sorted([p for p in root.iterdir() if p.is_dir()])
    if max_tracks is not None:
        tracks = tracks[:max_tracks]
    print(f"eval over {len(tracks)} tracks under {root}")

    aggregates = {st: {"sub": [], "ceil": [], "dd": []} for st in STEMS}
    skipped: List[Dict] = []
    n_chunks_seen = 0

    for t in tqdm(tracks, desc=desc):
        try:
            stems = _load_track_stems(t)
        except Exception as e:
            print(f"  !! {t.name}: load failed: {e}")
            skipped.append({"track": t.name, "reason": f"load failed: {e}"})
            continue
        sanity = _stem_sanity(stems)
        if sanity < sanity_thresh:
            print(f"  skipping {t.name}: stem identity {sanity:.1f} dB < {sanity_thresh}")
            skipped.append({"track": t.name, "reason": f"sanity={sanity:.2f}"})
            continue
        chunk_starts = _pick_chunks(stems, chunks_per_track, rng)
        if len(chunk_starts) == 0:
            skipped.append({"track": t.name, "reason": "no non-silent chunks"})
            continue
        per_stem = _process_track(model, stems, chunk_starts, device,
                                  batch_size, seed)
        for st in STEMS:
            aggregates[st]["sub"].extend(per_stem[st]["sub"])
            aggregates[st]["ceil"].extend(per_stem[st]["ceil"])
            aggregates[st]["dd"].extend(per_stem[st]["dd"])
        n_chunks_seen += len(chunk_starts)
        # Free track audio before next
        del stems

    summary: Dict[str, Dict[str, float]] = {}
    for st in STEMS:
        sub_vals = aggregates[st]["sub"]
        ceil_vals = aggregates[st]["ceil"]
        dd_vals = aggregates[st]["dd"]
        if not sub_vals:
            summary[st] = {"sdr_sub": 0.0, "sdr_ceil": 0.0, "sdr_dd": 0.0,
                           "gap": 0.0, "n": 0}
            continue
        s_sub = float(np.mean(sub_vals))
        s_ceil = float(np.mean(ceil_vals))
        summary[st] = {
            "sdr_sub": s_sub,
            "sdr_ceil": s_ceil,
            "sdr_dd": float(np.mean(dd_vals)),
            "gap": s_ceil - s_sub,
            "n": len(sub_vals),
        }

    all_sub = [v for st in STEMS for v in aggregates[st]["sub"]]
    all_ceil = [v for st in STEMS for v in aggregates[st]["ceil"]]
    all_dd = [v for st in STEMS for v in aggregates[st]["dd"]]
    if all_sub:
        a_sub = float(np.mean(all_sub))
        a_ceil = float(np.mean(all_ceil))
        summary["all"] = {
            "sdr_sub": a_sub,
            "sdr_ceil": a_ceil,
            "sdr_dd": float(np.mean(all_dd)),
            "gap": a_ceil - a_sub,
            "n": len(all_sub),
        }
    else:
        summary["all"] = {"sdr_sub": 0.0, "sdr_ceil": 0.0, "sdr_dd": 0.0,
                          "gap": 0.0, "n": 0}

    return summary, n_chunks_seen, skipped


def print_summary(summary: Dict, n_seen: int, skipped: List[Dict]):
    print(f"\n=== subtraction summary (n_chunks_seen={n_seen}, "
          f"{len(skipped)} tracks skipped) ===")
    print(f"{'stem':9s}  {'sdr_sub':>10s}  {'sdr_ceil':>10s}  {'sdr_dd':>10s}  "
          f"{'gap':>7s}  {'n':>6s}")
    for st in STEMS + ["all"]:
        s = summary[st]
        print(f"{st:9s}  {s['sdr_sub']:>+10.3f}  {s['sdr_ceil']:>+10.3f}  "
              f"{s.get('sdr_dd', 0.0):>+10.3f}  {s['gap']:>+7.3f}  {s['n']:>6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--musdb-dir", type=str, default="dataset/musdb18/test")
    ap.add_argument("--chunks-per-track", type=int, default=30)
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    device = get_device()

    model = Autoencoder(**build_model_config(cfg)).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    step = ck.get("global_step", -1)
    print(f"loaded {args.checkpoint} @ step {step}")

    summary, n_seen, skipped = run_eval(
        model=model,
        musdb_dir=args.musdb_dir,
        chunks_per_track=args.chunks_per_track,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        max_tracks=args.max_tracks,
        desc=f"subtraction/{Path(args.checkpoint).parent.name}",
    )

    print_summary(summary, n_seen, skipped)

    out_dict = {
        "checkpoint": args.checkpoint,
        "step": int(step) if isinstance(step, int) else -1,
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
