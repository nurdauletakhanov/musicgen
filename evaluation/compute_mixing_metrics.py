"""
Post-hoc mixing-equivariance metrics on the v1.1 / v2 test set.

For every val batch, generates a random fixed-point-free permutation, sets
α=0.5 (paper convention), computes:

  SDR_rec    — SI-SDR(g(f(x)), x)             reconstruction quality
  SDR_lin    — SI-SDR(g(z̄), g(f(x̄)))           linearity vs own recon ceiling (decode-vs-decode)
  SDR_lin_gt — SI-SDR(g(z̄), x̄)                linearity vs ground-truth mix (externally comparable)
  ℓ_lat    — ‖z̄ − f(x̄)‖² / ‖f(x̄)‖²            encoder linearity (normalized MSE)
  MixRate  — L_recon(g(z̄), x̄) / L_recon(g(f(x̄)), x̄)   decoder equivariance

All four reported per-source (musdb / maestro / fma / all). Output JSON.

Usage:
  python -m evaluation.compute_mixing_metrics \
      --config configs/experiments/v2/v2.1_decmix.yaml \
      --checkpoint checkpoints/v2.1-decmix/best.pth \
      --out evaluation/v2_metrics/v2.1-decmix_mixing.json

  # baseline
  python -m evaluation.compute_mixing_metrics \
      --config configs/experiments/v1/v1.1.yaml \
      --checkpoint checkpoints/v1.1/best.pth \
      --out evaluation/v2_metrics/v1.1_mixing.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.dataset import build_dataloaders
from models.autoencoder import Autoencoder
from training.config import build_model_config, get_device, load_config


def _si_sdr(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample SI-SDR (dB), returns [B] tensor."""
    x_hat = x_hat.float().reshape(x_hat.size(0), -1)
    x = x.float().reshape(x.size(0), -1)
    x = x - x.mean(dim=1, keepdim=True)
    x_hat = x_hat - x_hat.mean(dim=1, keepdim=True)
    alpha = (x_hat * x).sum(dim=1, keepdim=True) / (x.pow(2).sum(dim=1, keepdim=True) + eps)
    s_target = alpha * x
    e_noise = x_hat - s_target
    return 10.0 * torch.log10(
        (s_target.pow(2).sum(dim=1) + eps) / (e_noise.pow(2).sum(dim=1) + eps)
    )


def _no_fixed_point_perm(B: int, device, max_tries: int = 5) -> torch.Tensor:
    """Random permutation with no i -> i. Falls back to a cyclic shift."""
    arange = torch.arange(B, device=device)
    for _ in range(max_tries):
        perm = torch.randperm(B, device=device)
        if not (perm == arange).any():
            return perm
    return torch.roll(arange, shifts=max(1, B // 2))


def _per_sample_recon_loss(model, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Vectorized per-sample composite recon loss MR-STFT + Mel + L1, returns [B].

    Mirrors model.mrstft_loss + model.mel_loss + F.l1_loss but keeps the batch
    dimension instead of reducing to a scalar. Replaces the per-sample Python
    loop that dominated MixRate eval time.
    """
    if x.dim() == 3:
        x = x.squeeze(1)
    if x_hat.dim() == 3:
        x_hat = x_hat.squeeze(1)
    B = x.size(0)
    eps = float(model.eps)
    x_f32 = x.detach().float()
    x_hat_f32 = x_hat.float()

    # MR-STFT: per-sample SC + log-mag, averaged across FFT sizes.
    sc_per = x.new_zeros(B, dtype=torch.float32)
    mag_per = x.new_zeros(B, dtype=torch.float32)
    n_ffts = len(model.mrstft_ffts)
    for i, (n_fft, hop) in enumerate(zip(model.mrstft_ffts, model.mrstft_hops)):
        window = getattr(model, f"_mrstft_win_{i}")
        M = model._mrstft_mag(x_f32, n_fft, hop, window)        # [B, F, T]
        Mhat = model._mrstft_mag(x_hat_f32, n_fft, hop, window)
        num = torch.linalg.norm(M - Mhat, dim=(1, 2))           # [B]
        den = torch.linalg.norm(M, dim=(1, 2)).clamp(min=eps)   # [B]
        sc_per = sc_per + (num / den)
        mag_per = mag_per + (
            torch.log(Mhat + eps) - torch.log(M + eps)
        ).abs().mean(dim=(1, 2))                                # [B]
    sc_per = sc_per / n_ffts
    mag_per = mag_per / n_ffts
    mrstft_per = model.sc_weight * sc_per + model.mag_weight * mag_per  # [B]

    # Mel: per-sample log-mel L1.
    if model.mel_weight > 0.0:
        Xr = torch.stft(
            x_f32, n_fft=model.n_fft, hop_length=model.hop_length,
            win_length=model.win_length, window=model._mel_win,
            center=True, return_complex=True,
        )
        Xh = torch.stft(
            x_hat_f32, n_fft=model.n_fft, hop_length=model.hop_length,
            win_length=model.win_length, window=model._mel_win,
            center=True, return_complex=True,
        )
        mag_r = torch.sqrt(Xr.real ** 2 + Xr.imag ** 2 + eps)
        mag_h = torch.sqrt(Xh.real ** 2 + Xh.imag ** 2 + eps)
        mel_r = torch.einsum("nf,bft->bnt", model._mel_fb, mag_r)
        mel_h = torch.einsum("nf,bft->bnt", model._mel_fb, mag_h)
        mel_per = (
            torch.log(mel_h.clamp(min=eps)) - torch.log(mel_r.clamp(min=eps))
        ).abs().mean(dim=(1, 2))                                # [B]
    else:
        mel_per = x.new_zeros(B, dtype=torch.float32)

    # L1 on raw waveform, per-sample.
    l1_per = (x_hat_f32 - x_f32).abs().mean(dim=-1)             # [B]

    return mrstft_per + model.mel_weight * mel_per + l1_per


@torch.no_grad()
def _process_batch(
    model: Autoencoder,
    x_wave: torch.Tensor,
    sources: List[str],
    alpha: float = 0.5,
) -> Dict[str, List]:
    """Compute per-sample metrics for one batch. Returns lists of (source, value)."""
    device = x_wave.device
    if x_wave.dim() == 2:
        x_wave = x_wave.unsqueeze(1)
    B = x_wave.size(0)
    if B < 2:
        return {}
    perm = _no_fixed_point_perm(B, device)

    tgt = model.decoder.target_length
    x = x_wave[:, :, :tgt] if x_wave.size(2) > tgt else x_wave
    x_pair = x[perm]
    x_mix = alpha * x + (1.0 - alpha) * x_pair

    # Forward passes
    z = model.encoder(x)
    z_pair = z[perm]
    z_interp = alpha * z + (1.0 - alpha) * z_pair
    z_real = model.encoder(x_mix)

    g_recon, _ = model.decoder(z)            # g(f(x))      — for SDR_rec
    g_zbar, _ = model.decoder(z_interp)      # g(z̄)         — for SDR_lin, MixRate numerator
    g_zreal, _ = model.decoder(z_real)       # g(f(x̄))      — for SDR_lin, MixRate denominator

    # Per-sample SDR_rec on x (single-source recon)
    sdr_rec = _si_sdr(g_recon, x).cpu().tolist()

    # Per-sample SDR_lin between g(z̄) and g(f(x̄)) — decode-vs-decode,
    # equivariance relative to the model's own reconstruction ceiling.
    sdr_lin = _si_sdr(g_zbar, g_zreal).cpu().tolist()

    # Per-sample SDR_lin_gt between g(z̄) and the ground-truth mix x̄ —
    # the externally comparable variant (Torres et al. / M2L Table 3 style),
    # immune to inflation from a latent-insensitive decoder.
    sdr_lin_gt = _si_sdr(g_zbar, x_mix).cpu().tolist()

    # Per-sample ℓ_lat = ||z̄ - z_real||^2 / ||z_real||^2
    diff = (z_interp - z_real).reshape(B, -1)
    denom = z_real.reshape(B, -1)
    l_lat = (diff.pow(2).sum(dim=1) / denom.pow(2).sum(dim=1).clamp(min=1e-8)).cpu().tolist()

    # Per-sample MixRate = L_recon(g(z̄), x̄) / L_recon(g(f(x̄)), x̄)
    # Vectorized over the batch — replaces an earlier Python loop that
    # dominated runtime (~1 hr for one model). ~10× faster.
    loss_a = _per_sample_recon_loss(model, g_zbar, x_mix)   # [B]
    loss_b = _per_sample_recon_loss(model, g_zreal, x_mix)  # [B]
    mix_rate = (loss_a / loss_b.clamp(min=1e-8)).cpu().tolist()

    # Tag every sample with its source label
    out: Dict[str, List] = {
        "sdr_rec": list(zip(sources, sdr_rec)),
        "sdr_lin": list(zip(sources, sdr_lin)),
        "sdr_lin_gt": list(zip(sources, sdr_lin_gt)),
        "l_lat":   list(zip(sources, l_lat)),
        "mix_rate": list(zip(sources, mix_rate)),
    }
    return out


def _tally(records: List[tuple]) -> Dict[str, float]:
    """Group by source, return {source: mean, 'all': overall_mean}."""
    by_src: Dict[str, List[float]] = defaultdict(list)
    for src, val in records:
        by_src[src].append(val)
    out = {src: float(np.mean(vals)) for src, vals in by_src.items()}
    if records:
        out["all"] = float(np.mean([v for _, v in records]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Cap eval batches (smoke test)")
    ap.add_argument("--per-source", type=int, default=None,
                    help="Stratified subsample: this many chunks PER source "
                         "(fma/maestro/musdb), balanced. Preferred over "
                         "--max-batches for representative subsampling.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for permutation reproducibility")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    device = get_device()

    # Build model + load weights
    model = Autoencoder(**build_model_config(cfg)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("global_step", "?")
    print(f"loaded {args.checkpoint} @ step {step}")

    # Subsampling options (val set is source-contiguous, fma ~80%):
    #   --per-source N : balanced N chunks/source (preferred — representative).
    #   --max-batches  : proportional shuffled draw (fma-dominated; legacy).
    #   neither        : full deterministic eval.
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
    for bi, batch in enumerate(tqdm(val_loader, desc="mixing-metrics")):
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

    # Per-source sample counts — guards against a single-source subsample.
    from collections import Counter as _Counter
    src_counts = _Counter(s for s, _ in aggregates["sdr_lin"])
    print(f"per-source samples: {dict(src_counts)}  (n_seen={n_seen})")

    print("\n=== mixing metrics ===")
    for metric in ("sdr_rec", "sdr_lin", "sdr_lin_gt", "l_lat", "mix_rate"):
        line = f"  {metric:8s}"
        for src in sorted(summary[metric].keys()):
            line += f"  {src}={summary[metric][src]:+.4f}"
        print(line)

    out_dict = {
        "checkpoint": args.checkpoint,
        "step": int(step) if isinstance(step, int) else -1,
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
