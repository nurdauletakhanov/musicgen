"""
Overfit a single batch — fast sanity check for architecture + loss changes.

Usage:
    # Single phase:
    python overfit_one_batch.py --config <config> --steps 500

    # Two-phase (Option α — random cross-batch mixing):
    python overfit_one_batch.py --config <config> --phase1-steps 1000 --phase2-steps 500

    # Two-phase (Option β — same-track stem-pair mixing):
    python overfit_one_batch.py --config <config> --phase1-steps 1000 --phase2-steps 500 --stem-pairs
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.amp import autocast

from training.config import load_config, build_model_config
from models.autoencoder import Autoencoder
from data.dataloader import SingleStemDataset
from evaluation.utils import si_sdr

# Import archived stem-pair dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data", "_archive"))
from dataloader_stem_pairs import StemPairDataset


def compute_phase_err(est, ref, n_fft, hop_length, win_length, stft_window):
    hat_stft = torch.stft(
        est.squeeze(1), n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=stft_window,
        center=True, return_complex=True,
    )
    ref_stft = torch.stft(
        ref.squeeze(1), n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=stft_window,
        center=True, return_complex=True,
    )
    pd = torch.abs(hat_stft.angle() - ref_stft.angle())
    pd = torch.min(pd, 2 * 3.14159265 - pd)
    mag = ref_stft.abs()
    return float((pd * mag).sum() / mag.sum().clamp(min=1e-8) * 180 / 3.14159265)


def print_header():
    print(f"\n{'Step':>5} | {'Loss':>8} | {'Recon':>8} | {'MR':>7} | {'Mel':>7} | {'Mix':>7} | {'SI-SDR':>8} | {'Phase':>6}")
    print("-" * 85)


def eval_and_print(step, model, x_stft, x_wave, n_fft, hop_length, win_length, stft_window):
    model.eval()
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        _, comps, x_hat, _ = model(x_stft, x_wave, compute_mix_rate=True)

    tgt = model.decoder.target_length
    ref = x_wave[:, :, :tgt].float()
    est = x_hat.float()
    sdr = si_sdr(est, ref)
    phase_deg = compute_phase_err(est, ref, n_fft, hop_length, win_length, stft_window)

    print(
        f"{step:5d} | {comps['total']:8.4f} | "
        f"{comps['ReconSingle/Total']:8.4f} | "
        f"{comps['ReconSingle/MRSTFT']:7.4f} | "
        f"{comps['ReconSingle/Mel']:7.4f} | "
        f"{comps['DecodeMix/L1']:7.4f} | "
        f"{sdr:7.2f}dB | "
        f"{phase_deg:5.1f}°"
    )
    return sdr, phase_deg


def train_phase(model, optimizer, x_stft, x_wave, steps, print_every,
                n_fft, hop_length, win_length, stft_window):
    for step in range(1, steps + 1):
        model.train()
        with autocast("cuda", dtype=torch.bfloat16):
            total, comps, x_hat, _ = model(x_stft, x_wave)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % print_every == 0 or step == 1:
            eval_and_print(step, model, x_stft, x_wave,
                           n_fft, hop_length, win_length, stft_window)


def train_phase_stem_mix(model, optimizer, x_stft, x_wave, x_stft2, x_wave2,
                         mix_weight, steps, print_every,
                         n_fft, hop_length, win_length, stft_window):
    """Phase 2 with Option β: stem-pair mixing loss computed outside model.forward()."""
    for step in range(1, steps + 1):
        model.train()
        with autocast("cuda", dtype=torch.bfloat16):
            # Standard recon (mixing OFF inside model)
            total, comps, x_hat, _ = model(x_stft, x_wave)

            # Stem-pair mixing loss: encode both stems, interpolate, decode, compare
            B = x_stft.size(0)
            z1 = model.encoder(x_stft)
            z2 = model.encoder(x_stft2)

            alpha = torch.rand(B, 1, 1, device=x_stft.device)
            beta = 1.0 - alpha

            tgt = model.decoder.target_length
            w1 = x_wave[:, :, :tgt]
            w2 = x_wave2[:, :, :tgt]

            z_interp = alpha * z1 + beta * z2
            x_mix_wave = alpha * w1 + beta * w2

            x_interp, _ = model.decoder(z_interp)

            # Spectral + waveform L1 (same as Option α in autoencoder.py)
            mix_mr = model.mrstft_loss(x_interp, x_mix_wave)
            mix_mel = model.mel_loss(x_interp, x_mix_wave) if model.mel_weight > 0.0 else x_hat.new_tensor(0.0)
            mix_wav = F.l1_loss(x_interp, x_mix_wave)
            stem_mix_loss = model.mrstft_weight * mix_mr + model.mel_weight * mix_mel + mix_wav

            total = total + mix_weight * stem_mix_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % print_every == 0 or step == 1:
            # Eval prints recon metrics; show stem mix loss separately
            model.eval()
            with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                _, comps_eval, x_hat_eval, _ = model(x_stft, x_wave)

            tgt = model.decoder.target_length
            ref = x_wave[:, :, :tgt].float()
            est = x_hat_eval.float()
            sdr = si_sdr(est, ref)
            phase_deg = compute_phase_err(est, ref, n_fft, hop_length, win_length, stft_window)

            print(
                f"{step:5d} | {comps_eval['total'] + mix_weight * stem_mix_loss.detach().item():8.4f} | "
                f"{comps_eval['ReconSingle/Total']:8.4f} | "
                f"{comps_eval['ReconSingle/MRSTFT']:7.4f} | "
                f"{comps_eval['ReconSingle/Mel']:7.4f} | "
                f"{stem_mix_loss.detach().item():7.4f} | "
                f"{sdr:7.2f}dB | "
                f"{phase_deg:5.1f}°"
            )


def main():
    parser = argparse.ArgumentParser(description="Overfit one batch")
    parser.add_argument("--config", required=True, help="Path to experiment config")
    parser.add_argument("--num-chunks", type=int, default=12, help="Batch size (fixed batch)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--print-every", type=int, default=25)

    # Single-phase mode
    parser.add_argument("--steps", type=int, default=0, help="Single-phase steps")

    # Two-phase mode
    parser.add_argument("--phase1-steps", type=int, default=0, help="Phase 1: recon only")
    parser.add_argument("--phase2-steps", type=int, default=0, help="Phase 2: add mixing")
    parser.add_argument("--phase2-mix-weight", type=float, default=1.0, help="mixing weight for phase 2")
    parser.add_argument("--stem-pairs", action="store_true", help="Use Option β (same-track stem pairs) instead of Option α")
    parser.add_argument("--resume", type=str, default=None, help="Load model weights from checkpoint (skip phase 1)")

    args = parser.parse_args()

    single_phase = args.steps > 0
    two_phase = args.phase1_steps > 0 and args.phase2_steps > 0
    resume_only = args.resume is not None and not single_phase and not two_phase
    if not single_phase and not two_phase and not resume_only:
        parser.error("Use --steps, --phase1-steps + --phase2-steps, or --resume + --steps")

    cfg = load_config(args.config)
    data_cfg = cfg['data']
    stft_cfg = cfg.get('stft', {})
    mc = build_model_config(cfg)

    # For stem-pairs mode, mixing is handled outside model.forward()
    # For cross-batch mode or single-phase, keep config's mix weight
    if args.stem_pairs or two_phase:
        mc['decode_mix_weight'] = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(**mc).to(device)

    # Load checkpoint if provided
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        epoch = ckpt.get('epoch', '?')
        print(f"Loaded checkpoint: {args.resume} (epoch {epoch})")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params on {device}")
    mix_mode = "Option β (stem pairs)" if args.stem_pairs else "Option α (cross-batch)"
    print(f"Mix mode: {mix_mode}, mel_weight={model.mel_weight}")

    # Load data
    chunks_dir = data_cfg.get('chunks_dir') or data_cfg.get('musdb_chunks_dir')
    index_path = os.path.join(chunks_dir, "index.json")

    if args.stem_pairs:
        dataset = StemPairDataset(chunks_dir=chunks_dir, index_path=index_path, split="train")
        batch = [dataset[i] for i in range(min(args.num_chunks, len(dataset)))]
        x_stft = torch.stack([b["x_stft"] for b in batch]).to(device)
        x_wave = torch.stack([b["x_wave"] for b in batch]).to(device)
        x_stft2 = torch.stack([b["x_stft2"] for b in batch]).to(device)
        x_wave2 = torch.stack([b["x_wave2"] for b in batch]).to(device)
        print(f"Fixed batch: x_stft={tuple(x_stft.shape)}, x_wave={tuple(x_wave.shape)}")
        print(f"Stem pairs:  x_stft2={tuple(x_stft2.shape)}, x_wave2={tuple(x_wave2.shape)}")
    else:
        dataset = SingleStemDataset(chunks_dir=chunks_dir, index_path=index_path, split="train")
        batch = [dataset[i] for i in range(min(args.num_chunks, len(dataset)))]
        x_stft = torch.stack([b["x_stft"] for b in batch]).to(device)
        x_wave = torch.stack([b["x_wave"] for b in batch]).to(device)
        x_stft2 = x_wave2 = None
        print(f"Fixed batch: x_stft={tuple(x_stft.shape)}, x_wave={tuple(x_wave.shape)}")

    n_fft = stft_cfg.get('n_fft', 1024)
    hop_length = stft_cfg.get('hop_length', 256)
    win_length = stft_cfg.get('win_length', n_fft)
    stft_window = torch.hann_window(win_length).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if single_phase:
        if args.stem_pairs:
            print(f"\n=== Single phase: stem-pair mixing (weight={args.phase2_mix_weight}), {args.steps} steps ===")
            print_header()
            train_phase_stem_mix(model, optimizer, x_stft, x_wave, x_stft2, x_wave2,
                                 args.phase2_mix_weight, args.steps, args.print_every,
                                 n_fft, hop_length, win_length, stft_window)
        else:
            print_header()
            train_phase(model, optimizer, x_stft, x_wave, args.steps, args.print_every,
                        n_fft, hop_length, win_length, stft_window)
    else:
        # Phase 1: recon only
        print(f"\n=== Phase 1: recon only (mixing OFF), {args.phase1_steps} steps ===")
        print_header()
        train_phase(model, optimizer, x_stft, x_wave, args.phase1_steps, args.print_every,
                    n_fft, hop_length, win_length, stft_window)

        # Phase 2
        if args.stem_pairs:
            print(f"\n=== Phase 2: stem-pair mixing ON (weight={args.phase2_mix_weight}), {args.phase2_steps} steps ===")
            print_header()
            train_phase_stem_mix(model, optimizer, x_stft, x_wave, x_stft2, x_wave2,
                                 args.phase2_mix_weight, args.phase2_steps, args.print_every,
                                 n_fft, hop_length, win_length, stft_window)
        else:
            print(f"\n=== Phase 2: cross-batch mixing ON (weight={args.phase2_mix_weight}), {args.phase2_steps} steps ===")
            model.decode_mix_weight = args.phase2_mix_weight
            print(f"decode_mix_weight set to {model.decode_mix_weight}")
            print_header()
            train_phase(model, optimizer, x_stft, x_wave, args.phase2_steps, args.print_every,
                        n_fft, hop_length, win_length, stft_window)

    print("\nDone.")


if __name__ == "__main__":
    main()
