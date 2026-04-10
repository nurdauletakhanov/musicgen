"""Magnitude-vs-phase swap diagnostic for v11.

Question: which head of the v11 per-token MLP decoder is the bottleneck on
unseen data — magnitude, phase, or both?

Method: load a trained v11 checkpoint, run on unseen val chunks, then
reconstruct three ways via direct iSTFT:

  A. baseline:        pred_mag * exp(i * pred_phase)            -> what v11 outputs
  B. mag oracle:      target_mag * exp(i * pred_phase)          -> isolates magnitude error
  C. phase oracle:    pred_mag * exp(i * target_phase)          -> isolates phase error

For each mode, report mean SI-SDR + magnitude-weighted Phase angle on the
batch and write WAVs to results/diag_v11_swap/. Listen to all three.

Interpretation:
  - If B sounds great (and A bad), magnitude head is the bottleneck.
    -> v12 should keep per-token phase head, add cross-token magnitude head.
  - If C sounds great (and A bad), phase head is the bottleneck.
    -> the per-token MLP doesn't generalize off the overfit batch; encoder
       widening or a different phase head is needed.
  - If both B and C sound bad, encoder bottleneck — latent doesn't carry
    enough info for either head. Wider encoder.
  - If both sound great (unlikely), only A's combination is broken — joint
    training of the heads has gone wrong.

Usage:
    python diag_v11_swap.py --ckpt checkpoints/comp-22k-v11/best_model.pth
"""

import argparse
import math
import os

import torch

from training.config import load_config, build_model_config
from models.autoencoder import Autoencoder
from evaluation.utils import si_sdr


def load_val_batch(chunks_dir, num_chunks, device):
    val_dir = os.path.join(chunks_dir, "test")
    files = sorted(f for f in os.listdir(val_dir) if f.endswith(".pt"))
    assert files, f"No .pt files in {val_dir}"

    x_stft_list, x_wave_list = [], []
    fi = 0
    while len(x_stft_list) < num_chunks and fi < len(files):
        d = torch.load(os.path.join(val_dir, files[fi]),
                       map_location="cpu", weights_only=True)
        x_stft = d["x_stft"].float()
        x_wave = d["x_wave"].float()
        for i in range(x_stft.size(0)):
            if len(x_stft_list) >= num_chunks:
                break
            x_stft_list.append(x_stft[i])
            x_wave_list.append(x_wave[i])
        fi += 1

    x_stft = torch.stack(x_stft_list).to(device)
    x_wave = torch.stack(x_wave_list).unsqueeze(1).to(device)
    return x_stft, x_wave


def phase_error_deg(x_hat, x_wave, n_fft, hop, win, eps=1e-8):
    window = torch.hann_window(win, device=x_hat.device)
    Xh = torch.stft(x_hat.squeeze(1).float(), n_fft=n_fft, hop_length=hop,
                    win_length=win, window=window, center=True, return_complex=True)
    Xt = torch.stft(x_wave.squeeze(1).float(), n_fft=n_fft, hop_length=hop,
                    win_length=win, window=window, center=True, return_complex=True)
    diff = (Xh.angle() - Xt.angle()).abs()
    diff = torch.minimum(diff, 2 * math.pi - diff)
    mag = Xt.abs()
    return float((diff * mag).sum() / mag.sum().clamp(min=eps) * 180.0 / math.pi)


def istft_from_complex(C, n_fft, hop, win, length, device):
    window = torch.hann_window(win, device=device)
    return torch.istft(C, n_fft=n_fft, hop_length=hop, win_length=win,
                       window=window, center=True, length=length)


def report(name, x_hat, x_wave, n_fft, hop, win):
    sdr = si_sdr(x_hat.float(), x_wave.float())
    phase = phase_error_deg(x_hat, x_wave, n_fft, hop, win)
    rms = float(x_hat.float().pow(2).mean().sqrt())
    print(f"  {name:18s} SI-SDR={sdr:+7.2f} dB   Phase={phase:5.1f}deg   RMS={rms:.3f}")
    return sdr, phase


def save_wav(path, wave, sr):
    import numpy as np
    from scipy.io import wavfile
    w = wave.detach().cpu().float().clamp(-1, 1).numpy()
    wavfile.write(path, sr, (w * 32767).astype(np.int16))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/compression/comp_22k_v11.yaml")
    p.add_argument("--ckpt", default="checkpoints/comp-22k-v11/best_model.pth")
    p.add_argument("--num-chunks", type=int, default=8)
    p.add_argument("--out-dir", default="results/diag_v11_swap")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.config)
    mc = build_model_config(cfg)
    mc["dropout"] = 0.0

    model = Autoencoder(**mc).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_state = state.get("model_state_dict", state.get("model", state))
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  WARN missing keys: {len(missing)}")
    if unexpected:
        print(f"  WARN unexpected keys: {len(unexpected)}")
    model.eval()

    n_fft = mc["n_fft"]; hop = mc["hop_length"]; win = mc["win_length"]
    chunks_dir = cfg["data"]["chunks_dir"]
    sr = cfg["data"].get("sample_rate", 22050)

    print(f"Loading {args.num_chunks} val chunks from {chunks_dir}/test ...")
    x_stft, x_wave = load_val_batch(chunks_dir, args.num_chunks, device)
    print(f"  x_stft={tuple(x_stft.shape)}  x_wave={tuple(x_wave.shape)}")

    tgt_len = model.decoder.target_length
    if x_wave.size(-1) > tgt_len:
        x_wave = x_wave[..., :tgt_len]

    # Pad target STFT to encoder's internal length (matches forward())
    x_stft_aligned = model._pad_or_trunc_stft(x_stft, model.stft_t_full)

    with torch.no_grad():
        z = model.encoder(x_stft_aligned)
        x_hat_baseline, stft_pred = model.decoder(z)
        # stft_pred: [B, 2, F, T]  (real, imag)

        pred_complex = torch.complex(stft_pred[:, 0].float(), stft_pred[:, 1].float())
        tgt_complex = torch.complex(x_stft_aligned[:, 0].float(),
                                    x_stft_aligned[:, 1].float())

        pred_mag = pred_complex.abs()
        pred_phase = pred_complex.angle()
        tgt_mag = tgt_complex.abs()
        tgt_phase = tgt_complex.angle()

        eps = 1e-8

        # B: target mag * predicted phase
        B_complex = tgt_mag * torch.exp(1j * pred_phase)
        x_hat_B = istft_from_complex(B_complex, n_fft, hop, win, tgt_len, device).unsqueeze(1)

        # C: predicted mag * target phase
        C_complex = pred_mag * torch.exp(1j * tgt_phase)
        x_hat_C = istft_from_complex(C_complex, n_fft, hop, win, tgt_len, device).unsqueeze(1)

    print("\n=== Reconstruction modes ===")
    print("(measured against unseen val chunks)")
    report("A baseline",        x_hat_baseline, x_wave, n_fft, hop, win)
    report("B mag oracle",      x_hat_B,        x_wave, n_fft, hop, win)
    report("C phase oracle",    x_hat_C,        x_wave, n_fft, hop, win)

    # Save WAVs for listening
    os.makedirs(args.out_dir, exist_ok=True)
    n = min(4, x_wave.size(0))
    for i in range(n):
        save_wav(os.path.join(args.out_dir, f"{i:02d}_orig.wav"),  x_wave[i, 0], sr)
        save_wav(os.path.join(args.out_dir, f"{i:02d}_A_baseline.wav"),    x_hat_baseline[i, 0], sr)
        save_wav(os.path.join(args.out_dir, f"{i:02d}_B_mag_oracle.wav"),  x_hat_B[i, 0], sr)
        save_wav(os.path.join(args.out_dir, f"{i:02d}_C_phase_oracle.wav"), x_hat_C[i, 0], sr)
    print(f"\nWAVs written to {args.out_dir}/")
    print("Listen and compare A, B, C against orig. The one that sounds *good*")
    print("tells you which head the encoder still carries enough info for.")


if __name__ == "__main__":
    main()
