"""Test how denoising_steps affects M2L reconstruction quality."""
import math
import torch
from data.dataset import WaveformDataset
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
    val_ds = WaveformDataset("./chunks-44k-1s", split="test")

    ed = EncoderDecoder(device=device)
    print("checkpoint loaded; gen has", sum(p.numel() for p in ed.gen.parameters()), "params")

    # Take one chunk per source
    by_src = {}
    for i in range(min(200, len(val_ds))):
        item = val_ds[i]
        s = item["source"]
        if s not in by_src:
            by_src[s] = (i, item["x_wave"].squeeze(0).float())
        if len(by_src) == 3:
            break

    for steps in [1, 2, 4, 8, 16]:
        print(f"\n--- denoising_steps = {steps} ---")
        for src, (idx, x_wave) in by_src.items():
            z = ed.encode(x_wave.numpy())
            x_hat = ed.decode(z, denoising_steps=steps)
            if torch.is_tensor(x_hat):
                x_hat = x_hat.detach().cpu()
            x_hat_w = x_hat[0] if x_hat.dim() == 2 and x_hat.shape[0] <= 2 else x_hat.reshape(-1)
            n = min(x_hat_w.shape[0], x_wave.shape[0])
            sdr = si_sdr_db(x_hat_w[:n], x_wave[:n])
            # log-mag L1
            Xref = torch.stft(x_wave[:n], n_fft=1024, hop_length=256, win_length=1024,
                              window=torch.hann_window(1024), return_complex=True)
            Xa = torch.stft(x_hat_w[:n], n_fft=1024, hop_length=256, win_length=1024,
                            window=torch.hann_window(1024), return_complex=True)
            log_mag_l1 = (torch.log(Xa.abs() + 1e-8) - torch.log(Xref.abs() + 1e-8)).abs().mean().item()
            print(f"  {src:8s}  SDR={sdr:+7.2f} dB   log-mag L1={log_mag_l1:.4f}")


if __name__ == "__main__":
    main()
