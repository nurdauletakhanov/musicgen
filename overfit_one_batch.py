"""
Overfit-one-batch sanity check for the v11 per-token MLP STFT autoencoder.

Loads N chunks from the train split, trains the model for `steps` iterations
on that fixed batch, and reports SI-SDR + ComplexL1 + WaveL1 + Phase. Gate:
SI-SDR should reach > +15 dB on the training batch if the architecture and
loss path are sound. The diagnostic in diag_encoder_phase.py already showed
this combination (encoder + per-token MLP + complex_L1 + wave_L1) hits
+14.6 dB / 5° at step 5000 with no plateau in sight, so this script just
confirms it integrates cleanly through the production Autoencoder module.

Recon-only — no discriminator in v11 phase 1.

Usage:
    python overfit_one_batch.py
    python overfit_one_batch.py --config configs/experiments/compression/comp_22k_v11.yaml --steps 5000 --num-chunks 8
"""

import argparse
import math
import os

import torch

from training.config import load_config, build_model_config
from models.autoencoder import Autoencoder
from evaluation.utils import si_sdr


def load_fixed_batch(chunks_dir, num_chunks, device):
    """Load `num_chunks` chunks from the first train .pt file."""
    train_dir = os.path.join(chunks_dir, "train")
    files = sorted([f for f in os.listdir(train_dir) if f.endswith(".pt")])
    assert files, f"No .pt files in {train_dir}"

    x_stft_list, x_wave_list = [], []
    fi = 0
    while len(x_stft_list) < num_chunks and fi < len(files):
        d = torch.load(os.path.join(train_dir, files[fi]), map_location="cpu", weights_only=True)
        x_stft = d["x_stft"].float()    # [N, 2, F, T]
        x_wave = d["x_wave"].float()    # [N, L]
        for i in range(x_stft.size(0)):
            if len(x_stft_list) >= num_chunks:
                break
            x_stft_list.append(x_stft[i])
            x_wave_list.append(x_wave[i])
        fi += 1

    x_stft = torch.stack(x_stft_list).to(device)               # [B, 2, F, T]
    x_wave = torch.stack(x_wave_list).unsqueeze(1).to(device)  # [B, 1, L]
    return x_stft, x_wave


def phase_error_deg(x_hat, x_wave, n_fft=1024, hop=256, win=1024, eps=1e-8):
    """Mean angular error in degrees, magnitude-weighted."""
    window = torch.hann_window(win, device=x_hat.device)
    Xh = torch.stft(x_hat.squeeze(1).float(), n_fft=n_fft, hop_length=hop,
                    win_length=win, window=window, center=True, return_complex=True)
    Xt = torch.stft(x_wave.squeeze(1).float(), n_fft=n_fft, hop_length=hop,
                    win_length=win, window=window, center=True, return_complex=True)
    diff = (Xh.angle() - Xt.angle()).abs()
    diff = torch.minimum(diff, 2 * math.pi - diff)
    mag = Xt.abs()
    return float((diff * mag).sum() / mag.sum().clamp(min=eps) * 180.0 / math.pi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="configs/experiments/compression/comp_22k_v15.yaml")
    p.add_argument("--num-chunks", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wave-l1", type=float, default=1.0,
                   help="Extra waveform L1 weight added on top of the config's "
                        "loss. Overfit with mel+MRSTFT only has no phase "
                        "signal; wave_l1 provides one. Set 0 to disable.")
    p.add_argument("--only-wl1", action="store_true",
                   help="Zero out mel/mrstft in the model so only --wave-l1 "
                        "drives training. Isolates the phase signal from the "
                        "mel magnitude attractor (which is 15x larger).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.config)
    model_cfg = build_model_config(cfg)
    model_cfg["dropout"] = 0.0  # no dropout for overfit

    print(f"Model config: d_model={model_cfg['d_model']} n_layers={model_cfg['n_layers']} "
          f"target_length={model_cfg['target_length']} num_segments={model_cfg['num_segments']}")
    print(f"Recon recipe: mel_l1={model_cfg['mel_l1_weight']}, "
          f"mrstft_mag={model_cfg['mrstft_mag_l1_weight']}, "
          f"latent_l2={model_cfg['latent_l2_weight']}")

    if args.only_wl1:
        model_cfg["mel_l1_weight"] = 0.0
        model_cfg["mrstft_mag_l1_weight"] = 0.0
        print("  --only-wl1: zeroed mel_l1 and mrstft_mag_l1")

    model = Autoencoder(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_enc = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    n_dec = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    print(f"num_parameters: {n_params:.2f} M  "
          f"(encoder: {n_enc:.2f} M, decoder: {n_dec:.2f} M)")

    chunks_dir = cfg["data"]["chunks_dir"]
    print(f"Loading {args.num_chunks} chunks from {chunks_dir}/train ...")
    x_stft, x_wave = load_fixed_batch(chunks_dir, args.num_chunks, device)
    print(f"  x_stft={tuple(x_stft.shape)}  x_wave={tuple(x_wave.shape)}")

    # Truncate x_wave to decoder target_length (preprocessing pads)
    tgt = model.decoder.target_length
    if x_wave.size(-1) > tgt:
        x_wave = x_wave[..., :tgt]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.0)

    print(f"Aux wave_l1 (overfit-only phase signal): {args.wave_l1}")

    model.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        total, comps, x_hat, _ = model(x_stft, x_wave)
        if args.wave_l1 > 0.0:
            wl1 = torch.nn.functional.l1_loss(x_hat, x_wave)
            total = total + args.wave_l1 * wl1
            comps['ReconSingle/WaveL1'] = float(wl1.detach())
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                sdr = si_sdr(x_hat.float(), x_wave.float())
                phase = phase_error_deg(x_hat, x_wave,
                                        n_fft=model_cfg['n_fft'],
                                        hop=model_cfg['hop_length'],
                                        win=model_cfg['win_length'])
            print(f"step {step:4d}  total={float(total.detach()):8.4f}  "
                  f"WL1={comps['ReconSingle/WaveL1']:.4f}  "
                  f"Mel={comps['ReconSingle/MelL1']:.4f}  "
                  f"MRSTFT={comps['ReconSingle/MRSTFTMag']:.4f}  "
                  f"SI-SDR={sdr:+7.2f} dB  Phase={phase:5.1f}deg")

    # Final eval
    model.eval()
    with torch.no_grad():
        total, comps, x_hat, _ = model(x_stft, x_wave)
        sdr = si_sdr(x_hat.float(), x_wave.float())
        phase = phase_error_deg(x_hat, x_wave,
                                n_fft=model_cfg['n_fft'],
                                hop=model_cfg['hop_length'],
                                win=model_cfg['win_length'])
    print(f"\nFINAL  total={float(total):.4f}  SI-SDR={sdr:+.2f} dB  Phase={phase:.1f}deg")
    print(f"GATE: SI-SDR > +15 dB -> {'PASS' if sdr > 15.0 else 'FAIL'}")
    print(f"GATE: Phase  < 30 deg -> {'PASS' if phase < 30.0 else 'FAIL'}")

    # Save audio for listening
    out_dir = "results/overfit_v15"
    os.makedirs(out_dir, exist_ok=True)
    from scipy.io import wavfile
    import numpy as np
    sr = cfg["data"].get("sample_rate", 22050)
    for i in range(min(4, x_hat.size(0))):
        orig = x_wave[i, 0].cpu().float().numpy()
        recon = x_hat[i, 0].cpu().float().clamp(-1, 1).numpy()
        wavfile.write(os.path.join(out_dir, f"{i:02d}_orig.wav"),
                      sr, (orig * 32767).astype(np.int16))
        wavfile.write(os.path.join(out_dir, f"{i:02d}_recon.wav"),
                      sr, (recon * 32767).astype(np.int16))
    print(f"Audio saved to {out_dir}/")


if __name__ == "__main__":
    main()
