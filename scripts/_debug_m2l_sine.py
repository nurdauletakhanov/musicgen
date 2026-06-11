"""Sanity test: feed M2L a pure sine wave, expect near-perfect reconstruction
of the magnitude (sine = single STFT bin)."""
import math
import torch
import numpy as np
from training.config import get_device
from music2latent.inference import EncoderDecoder


def si_sdr_db(x_hat, x, eps=1e-8):
    x_hat = x_hat.float().reshape(-1)
    x = x.float().reshape(-1)
    x = x - x.mean()
    x_hat = x_hat - x_hat.mean()
    alpha = (x_hat * x).sum() / (x.pow(2).sum() + eps)
    s_target = alpha * x
    e_noise = x_hat - s_target
    return float(10.0 * math.log10((s_target.pow(2).sum().item() + eps) / (e_noise.pow(2).sum().item() + eps)))


def main():
    device = get_device()
    sr = 44100

    ed = EncoderDecoder(device=device)

    # 5 s of 440 Hz sine at 44.1 kHz
    for dur, freq in [(5.0, 440.0), (1.0, 440.0), (10.0, 220.0)]:
        n = int(dur * sr)
        t = np.arange(n) / sr
        wav = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        z = ed.encode(wav)
        x_hat = ed.decode(z)
        if torch.is_tensor(x_hat):
            x_hat = x_hat.detach().cpu()
        x_hat_w = x_hat[0] if x_hat.dim() == 2 and x_hat.shape[0] <= 2 else x_hat.reshape(-1)
        m = min(x_hat_w.shape[0], len(wav))
        sdr = si_sdr_db(x_hat_w[:m], torch.from_numpy(wav[:m]))

        # log-mag L1 (magnitude reconstruction)
        Xref = torch.stft(torch.from_numpy(wav[:m]), n_fft=2048, hop_length=512,
                          win_length=2048, window=torch.hann_window(2048), return_complex=True)
        Xa = torch.stft(x_hat_w[:m], n_fft=2048, hop_length=512,
                        win_length=2048, window=torch.hann_window(2048), return_complex=True)
        log_mag_l1 = (torch.log(Xa.abs() + 1e-8) - torch.log(Xref.abs() + 1e-8)).abs().mean().item()
        print(f"sine {freq}Hz {dur}s ({n} samples) -> hat shape {x_hat_w.shape}  "
              f"SDR={sdr:+7.2f} dB  log-mag L1={log_mag_l1:.4f}  "
              f"hat-peak={x_hat_w.abs().max().item():.3f}")


def save_recon_wavs():
    """Encode-decode 3 real chunks and save (ref, recon) pairs as WAVs so the
    user can listen and judge perceptual quality directly."""
    import os
    import soundfile as sf
    from data.dataset import WaveformDataset
    device = get_device()
    ed = EncoderDecoder(device=device)
    val_ds = WaveformDataset("./chunks-44k-1s", split="test")

    out_dir = "scripts/_m2l_recon_samples"
    os.makedirs(out_dir, exist_ok=True)

    by_src = {}
    # Walk the dataset until we have one chunk for each source
    for i in range(len(val_ds)):
        item = val_ds[i]
        s = item["source"]
        if s not in by_src:
            by_src[s] = (i, item["x_wave"].squeeze(0).float())
        if len(by_src) == 3:
            break

    for src, (idx, x_wave) in by_src.items():
        # 8 s of consecutive chunks for a less choppy listen
        chunks = [x_wave]
        cur = idx
        f, _ = val_ds._lookup(cur)
        for k in range(1, 8):
            j = cur + k
            if j >= f["end"]:
                break
            chunks.append(val_ds[j]["x_wave"].squeeze(0).float())
        long_wave = torch.cat(chunks, dim=0).numpy()

        z = ed.encode(long_wave)
        x_hat = ed.decode(z, denoising_steps=1)
        if torch.is_tensor(x_hat):
            x_hat = x_hat.detach().cpu()
        x_hat_np = (x_hat[0] if x_hat.dim() == 2 and x_hat.shape[0] <= 2 else x_hat.reshape(-1)).numpy()
        n = min(len(x_hat_np), len(long_wave))
        sf.write(f"{out_dir}/{src}_ref.wav",   long_wave[:n], 44100)
        sf.write(f"{out_dir}/{src}_recon.wav", x_hat_np[:n],  44100)
        print(f"wrote {src}_ref.wav / {src}_recon.wav  ({n} samples = {n/44100:.2f}s)")


if __name__ == "__main__":
    main()
    print("\n----- saving recon WAVs for listening -----")
    save_recon_wavs()
