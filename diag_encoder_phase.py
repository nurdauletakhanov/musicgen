"""
Diagnostic: does the v10 encoder preserve phase information in the latent?

Setup:
    - Use the v10 encoder unchanged: 2D conv stack on complex STFT [B, 2, F, T]
      -> latent [B, 29, 96]
    - Pair it with a MINIMUM-CAPACITY decoder: per-token linear projection
      96 -> 2*513 = 1026 channels, reshape to [B, 2, 513, 29], bilinear
      upsample in time to [B, 2, 513, T]. No conv stack, no smoothing.
    - Loss: direct complex L1 on (real, imag) of the predicted STFT vs target.
      This is the most phase-aware loss possible.
    - Train end-to-end on 8 fixed chunks for N steps.

Interpretation:
    - If Phase drops below ~30 deg and SI-SDR > +10 dB: the encoder CAN encode
      phase. The bottleneck is downstream (decoder is too weak / destroying
      phase). Action: fix the decoder.
    - If Phase stays near 90 deg: the encoder architecture (strided 2D convs
      with 24x compression) cannot retain phase. Action: change the encoder
      (lower compression, or 1D wave encoder, or different parameterization).

This is a structural test, not a training run. ~1-2 minutes on the 5090.
"""

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import load_config, build_model_config
from models.encoder import Encoder
from evaluation.utils import si_sdr


def load_fixed_batch(chunks_dir, num_chunks, device):
    train_dir = os.path.join(chunks_dir, "train")
    files = sorted([f for f in os.listdir(train_dir) if f.endswith(".pt")])
    x_stft_list, x_wave_list = [], []
    fi = 0
    while len(x_stft_list) < num_chunks and fi < len(files):
        d = torch.load(os.path.join(train_dir, files[fi]), map_location="cpu", weights_only=True)
        for i in range(d["x_stft"].size(0)):
            if len(x_stft_list) >= num_chunks:
                break
            x_stft_list.append(d["x_stft"][i].float())
            x_wave_list.append(d["x_wave"][i].float())
        fi += 1
    x_stft = torch.stack(x_stft_list).to(device)
    x_wave = torch.stack(x_wave_list).unsqueeze(1).to(device)
    return x_stft, x_wave


def phase_error_deg(stft_pred, stft_target, eps=1e-8):
    """Magnitude-weighted phase error in degrees, computed on STFT directly."""
    # stft_pred, stft_target: [B, 2, F, T]
    pr = stft_pred[:, 0].float()
    pi = stft_pred[:, 1].float()
    tr = stft_target[:, 0].float()
    ti = stft_target[:, 1].float()
    pred_phase = torch.atan2(pi, pr)
    tgt_phase = torch.atan2(ti, tr)
    diff = (pred_phase - tgt_phase).abs()
    diff = torch.minimum(diff, 2 * math.pi - diff)
    mag = torch.sqrt(tr ** 2 + ti ** 2 + eps)
    return float((diff * mag).sum() / mag.sum().clamp(min=eps) * 180.0 / math.pi)


class TinyDecoder(nn.Module):
    """Minimum-capacity decoder: per-token (linear or MLP) + bilinear time upsample.

    Input:  [B, S, D] latent
    Output: [B, 2, F, T_full] complex STFT

    Three modes:
        - 'linear': Linear D -> 2*F, then bilinear upsample S -> t_full
                    (per-token 96-dim subspace; bilinear interpolates phase = bad)
        - 'mlp':    Linear D -> hidden -> GELU -> Linear hidden -> 2*F, then bilinear
                    (more capacity to untangle the latent; still bilinear-limited)
        - 'direct': Linear D -> frames_per_token * 2 * F. Each latent token directly
                    emits its own block of STFT frames. No interpolation, no spatial
                    mixing across tokens. This is the true "can the latent encode
                    phase" test — bilinear cannot reconstruct oscillating phase by
                    construction, so 'linear'/'mlp' confound encoder-vs-upsample.
    """

    def __init__(self, d_model, n_freq_bins, num_segments, t_full,
                 mode='linear', hidden=512):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.num_segments = num_segments
        self.t_full = t_full
        self.mode = mode
        if mode == 'linear':
            self.proj = nn.Linear(d_model, 2 * n_freq_bins)
        elif mode == 'mlp':
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2 * n_freq_bins),
            )
        elif mode == 'direct':
            assert t_full % num_segments == 0, \
                f"'direct' mode requires t_full ({t_full}) divisible by num_segments ({num_segments})"
            self.frames_per_token = t_full // num_segments
            self.proj = nn.Linear(d_model, self.frames_per_token * 2 * n_freq_bins)
        elif mode == 'direct_mlp':
            assert t_full % num_segments == 0, \
                f"'direct_mlp' mode requires t_full ({t_full}) divisible by num_segments ({num_segments})"
            self.frames_per_token = t_full // num_segments
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.frames_per_token * 2 * n_freq_bins),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, z):
        B, S, D = z.shape
        if self.mode in ('direct', 'direct_mlp'):
            h = self.proj(z)                                       # [B, S, fpt*2*F]
            h = h.view(B, S, self.frames_per_token, 2, self.n_freq_bins)
            # [B, S, fpt, 2, F] -> [B, 2, F, S, fpt] -> [B, 2, F, S*fpt]
            h = h.permute(0, 3, 4, 1, 2).contiguous()
            h = h.view(B, 2, self.n_freq_bins, S * self.frames_per_token)
            return h
        h = self.proj(z)                                  # [B, S, 2*F]
        h = h.view(B, S, 2, self.n_freq_bins)             # [B, S, 2, F]
        h = h.permute(0, 2, 3, 1).contiguous()            # [B, 2, F, S]
        h = F.interpolate(h, size=(self.n_freq_bins, self.t_full),
                          mode='bilinear', align_corners=False)
        return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="configs/experiments/compression/comp_22k_v10.yaml")
    p.add_argument("--num-chunks", type=int, default=8)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--decoder", type=str, default="linear",
                   choices=["linear", "mlp", "direct", "direct_mlp"],
                   help="'linear' = per-token linear + bilinear (bilinear is phase-blind). "
                        "'mlp' = per-token MLP + bilinear. "
                        "'direct' = per-token Linear D -> (fpt*2*F), no interpolation. "
                        "'direct_mlp' = per-token MLP D -> hidden -> (fpt*2*F), no interpolation. "
                        "More capacity per token for the same structure.")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--d-model", type=int, default=None,
                   help="Override encoder d_model (transformer width + freq_collapse output). "
                        "Default = config value (96).")
    p.add_argument("--encoder-channels", type=str, default=None,
                   help="Override encoder CNN channels as comma-separated ints, e.g. "
                        "'64,128,128,128,128'. Must match len(freq_strides)=5.")
    p.add_argument("--wave-l1", type=float, default=0.0,
                   help="Weight for waveform-domain L1 (iSTFT(pred) vs x_wave). "
                        "0 = off (pure complex L1). >0 adds the term. Try 1.0.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.config)
    mc = build_model_config(cfg)

    n_fft = mc['n_fft']
    hop = mc['hop_length']
    win = mc['win_length']
    sr = mc['sample_rate']
    n_freq = mc['n_freq_bins']
    num_segments = mc['num_segments']
    target_length = mc['target_length']

    # Apply encoder overrides
    if args.d_model is not None:
        mc['d_model'] = args.d_model
        print(f"[override] d_model = {args.d_model}")
    if args.encoder_channels is not None:
        ec = [int(x) for x in args.encoder_channels.split(",")]
        assert len(ec) == len(mc['freq_strides']), \
            f"--encoder-channels has {len(ec)} entries, freq_strides has {len(mc['freq_strides'])}"
        mc['encoder_channels'] = ec
        print(f"[override] encoder_channels = {ec}")

    # Compute STFT time frames for the target STFT
    # encoder pads/truncates to required_T = num_segments * total_time_stride
    total_time_stride = 1
    for s in mc['time_strides']:
        total_time_stride *= s
    t_full = num_segments * total_time_stride
    print(f"Encoder: n_freq={n_freq} num_segments={num_segments} t_full={t_full} "
          f"compression={2 * n_freq * t_full // (num_segments * mc['d_model'])}x")

    encoder = Encoder(
        d_model=mc['d_model'],
        n_heads=mc['n_heads'],
        n_layers=mc['n_layers'],
        num_segments=num_segments,
        n_freq_bins=n_freq,
        dropout=0.0,
        encoder_channels=mc['encoder_channels'],
        freq_strides=mc['freq_strides'],
        time_strides=mc['time_strides'],
    ).to(device)

    decoder = TinyDecoder(
        d_model=mc['d_model'],
        n_freq_bins=n_freq,
        num_segments=num_segments,
        t_full=t_full,
        mode=args.decoder,
        hidden=args.hidden,
    ).to(device)
    print(f"tiny decoder mode: {args.decoder}" + (f" (hidden={args.hidden})" if args.decoder == 'mlp' else ""))

    n_enc = sum(p.numel() for p in encoder.parameters()) / 1e6
    n_dec = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"encoder params: {n_enc:.2f} M  |  tiny decoder params: {n_dec:.2f} M")

    chunks_dir = cfg["data"]["chunks_dir"]
    x_stft, x_wave = load_fixed_batch(chunks_dir, args.num_chunks, device)
    print(f"x_stft={tuple(x_stft.shape)}  x_wave={tuple(x_wave.shape)}")

    # Truncate target STFT time dim to t_full to match encoder's padding
    if x_stft.size(-1) > t_full:
        x_stft = x_stft[..., :t_full]
    elif x_stft.size(-1) < t_full:
        x_stft = F.pad(x_stft, (0, t_full - x_stft.size(-1)))

    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.8, 0.99), weight_decay=0.0)

    window = torch.hann_window(win, device=device)

    encoder.train()
    decoder.train()
    if args.wave_l1 > 0.0:
        print(f"loss: complex_L1 + {args.wave_l1} * wave_L1(iSTFT(pred), x_wave)")
    else:
        print(f"loss: complex_L1 only")
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        z = encoder(x_stft)
        stft_pred = decoder(z)
        # Direct complex L1 (per-bin, dense, phase-aware)
        cl1 = F.l1_loss(stft_pred, x_stft)
        loss = cl1
        wl1_val = 0.0
        if args.wave_l1 > 0.0:
            pred_c = torch.complex(stft_pred[:, 0].float(), stft_pred[:, 1].float())
            wav_pred = torch.istft(pred_c, n_fft=n_fft, hop_length=hop,
                                   win_length=win, window=window, center=True,
                                   length=target_length)
            wav_tgt = x_wave[:, 0, :target_length].float()
            wl1 = F.l1_loss(wav_pred, wav_tgt)
            loss = loss + args.wave_l1 * wl1
            wl1_val = float(wl1.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                phase = phase_error_deg(stft_pred, x_stft)
                # iSTFT for SI-SDR
                pred_c = torch.complex(stft_pred[:, 0].float(), stft_pred[:, 1].float())
                wav = torch.istft(pred_c, n_fft=n_fft, hop_length=hop,
                                  win_length=win, window=window, center=True,
                                  length=target_length)
                wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(1)
                ref = x_wave[..., :target_length]
                sdr = si_sdr(wav.float(), ref.float())
                # Magnitude error (sanity)
                pred_mag = torch.sqrt(stft_pred[:, 0] ** 2 + stft_pred[:, 1] ** 2 + 1e-8)
                tgt_mag = torch.sqrt(x_stft[:, 0] ** 2 + x_stft[:, 1] ** 2 + 1e-8)
                mag_l1 = F.l1_loss(pred_mag, tgt_mag).item()
            wl1_str = f"  WL1={wl1_val:.5f}" if args.wave_l1 > 0.0 else ""
            print(f"step {step:4d}  CL1={float(cl1.detach()):.5f}{wl1_str}  "
                  f"MagL1={mag_l1:.5f}  SI-SDR={sdr:+7.2f} dB  Phase={phase:5.1f}deg")

    print("\n--- Diagnosis ---")
    if phase < 30.0 and sdr > 10.0:
        print(f"PHASE FITS (Phase={phase:.1f}deg, SI-SDR={sdr:+.2f} dB)")
        print("=> The encoder CAN encode phase. Bottleneck is downstream.")
        print("=> Action: increase decoder capacity or use a different decoder.")
    elif phase < 60.0:
        print(f"PHASE PARTIAL (Phase={phase:.1f}deg, SI-SDR={sdr:+.2f} dB)")
        print("=> Encoder leaks phase but slowly. Bottleneck is partially encoder.")
        print("=> Action: try lower compression or different encoder parameterization.")
    else:
        print(f"PHASE STUCK (Phase={phase:.1f}deg, SI-SDR={sdr:+.2f} dB)")
        print("=> The encoder is destroying phase information.")
        print("=> Action: change the encoder (1D wave conv, lower compression, etc.)")


if __name__ == "__main__":
    main()
