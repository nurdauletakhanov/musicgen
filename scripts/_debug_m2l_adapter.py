"""Debug script: compare M2L public inference vs our adapter on the same chunk.

Loads one chunk from chunks-44k-1s/test, runs:
  (A) music2latent.inference.EncoderDecoder.encode/decode (the canonical path)
  (B) M2LAutoencoderAdapter.encoder/decoder (our adapter)

Prints SDR(recon, ref) for each, and the latent shape/dtype/range. If (A) works
and (B) does not, the adapter is wrong; if both fail, something else is off.
"""
import math
import torch
from data.dataset import WaveformDataset
from training.config import get_device
from music2latent.inference import EncoderDecoder
from evaluation.m2l_adapter import M2LAutoencoderAdapter


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
    print("device", device)

    val_ds = WaveformDataset("./chunks-44k-1s", split="test")
    item = val_ds[0]
    x_wave = item["x_wave"]  # [1, 44100] fp32, mono
    if x_wave.dim() == 2:
        x_wave = x_wave.squeeze(0)  # [44100]
    print("input chunk shape", x_wave.shape, "dtype", x_wave.dtype, "src", item["source"])
    print("input rms", x_wave.float().pow(2).mean().sqrt().item())

    # ------------------ (A) Public M2L inference path -------------------
    ed = EncoderDecoder(device=device)
    print("EncoderDecoder loaded.")
    z_pub = ed.encode(x_wave.numpy())  # numpy [1, dim, length]
    print("encode returned", type(z_pub), getattr(z_pub, "shape", None),
          "dtype", getattr(z_pub, "dtype", None))
    if torch.is_tensor(z_pub):
        z_pub_arr = z_pub.detach().cpu().numpy()
    else:
        z_pub_arr = z_pub
    print("z_pub stats:  min", z_pub_arr.min(), "max", z_pub_arr.max(),
          "mean", z_pub_arr.mean(), "std", z_pub_arr.std())

    x_hat_pub = ed.decode(z_pub)
    if torch.is_tensor(x_hat_pub):
        x_hat_pub = x_hat_pub.detach().cpu()
    else:
        x_hat_pub = torch.from_numpy(x_hat_pub)
    print("x_hat_pub shape", x_hat_pub.shape, "dtype", x_hat_pub.dtype)
    # Output is [audio_channels, samples] for mono => [1, ~42496]. Take channel 0.
    if x_hat_pub.dim() == 2 and x_hat_pub.shape[0] <= 2:
        x_hat_pub_w = x_hat_pub[0]
    else:
        x_hat_pub_w = x_hat_pub.reshape(-1)
    n = min(x_hat_pub_w.shape[0], x_wave.shape[0])
    print("compare lengths: x_hat", x_hat_pub_w.shape, "x_wave", x_wave.shape, "n=", n)
    sdr_pub = si_sdr_db(x_hat_pub_w[:n], x_wave[:n])
    print("PUBLIC INFERENCE SDR =", sdr_pub, "dB")
    # Sanity: peak amplitude
    print("  x_hat_pub peak", x_hat_pub_w.abs().max().item(),
          " x_wave peak", x_wave.abs().max().item())

    # ------------------ (B) Our adapter ----------------------------------
    adapter = M2LAutoencoderAdapter(device=device).to(device)
    adapter.eval()
    x_in = x_wave.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 44100]
    with torch.no_grad():
        z_ad = adapter.encoder(x_in[:, :, :adapter.decoder.target_length])
        print("adapter z shape", z_ad.shape, "dtype", z_ad.dtype,
              "min", z_ad.min().item(), "max", z_ad.max().item(),
              "mean", z_ad.mean().item(), "std", z_ad.std().item())
        x_hat_ad, _ = adapter.decoder(z_ad)
    print("adapter x_hat shape", x_hat_ad.shape)
    sdr_ad = si_sdr_db(x_hat_ad.squeeze().cpu(),
                       x_wave[:adapter.decoder.target_length])
    print("ADAPTER SDR =", sdr_ad, "dB")


def main2_phase_test():
    """Test: is the bad SDR caused by phase randomness from the noise init?

    Decode the SAME latent twice with two different random noises (default
    behavior) vs the SAME noise (deterministic). Compare SDRs.
    """
    device = get_device()
    val_ds = WaveformDataset("./chunks-44k-1s", split="test")
    x_wave = val_ds[0]["x_wave"].squeeze(0).float()  # [44100]

    from music2latent import hparams as hp
    from music2latent.audio import to_representation_encoder, to_waveform
    from music2latent.inference import EncoderDecoder
    ed = EncoderDecoder(device=device)
    gen = ed.gen
    gen.eval()

    target_length = 42496
    x_in = x_wave[:target_length].to(device)
    repr_enc = to_representation_encoder(x_in.unsqueeze(0))
    with torch.no_grad():
        z = gen.encoder(repr_enc)
    print("z shape", z.shape, "raw tanh stats: min", z.min().item(), "max", z.max().item())

    downscaling = 2 ** hp.freq_downsample_list.count(0)
    T_stft = z.shape[-1] * downscaling

    def decode_with_noise(noise):
        with torch.no_grad():
            pyr = gen.decoder(z)
            x_stft = gen(z, noise, sigma=hp.sigma_max, pyramid_latents=pyr)
            return to_waveform(x_stft).squeeze(0).cpu()

    g = torch.Generator(device=device).manual_seed(42)
    noise_a = torch.randn((1, hp.data_channels, hp.hop*2, T_stft), device=device, generator=g) * hp.sigma_max
    g.manual_seed(42)
    noise_a2 = torch.randn((1, hp.data_channels, hp.hop*2, T_stft), device=device, generator=g) * hp.sigma_max
    noise_b = torch.randn((1, hp.data_channels, hp.hop*2, T_stft), device=device) * hp.sigma_max

    x_hat_a = decode_with_noise(noise_a)
    x_hat_a_again = decode_with_noise(noise_a2)
    x_hat_b = decode_with_noise(noise_b)

    print("SDR(decode_a, original) =", si_sdr_db(x_hat_a, x_wave[:target_length].cpu()))
    print("SDR(decode_a_again, decode_a) =", si_sdr_db(x_hat_a_again, x_hat_a),
          "  (should be inf — deterministic with same seed)")
    print("SDR(decode_b, decode_a) =", si_sdr_db(x_hat_b, x_hat_a),
          "  (random noise — measures phase variance)")
    print("SDR(decode_b, original) =", si_sdr_db(x_hat_b, x_wave[:target_length].cpu()))

    # Also: STFT-magnitude error to confirm the model IS reconstructing magnitude
    Xref = torch.stft(x_wave[:target_length].cpu(), n_fft=1024, hop_length=256,
                      win_length=1024, window=torch.hann_window(1024), return_complex=True)
    Xa = torch.stft(x_hat_a, n_fft=1024, hop_length=256,
                    win_length=1024, window=torch.hann_window(1024), return_complex=True)
    mag_ref = Xref.abs() + 1e-8
    mag_a = Xa.abs() + 1e-8
    log_mag_l1 = (torch.log(mag_a) - torch.log(mag_ref)).abs().mean().item()
    print("log-mag L1(decode_a, original) =", log_mag_l1,
          "  (low = good magnitude reconstruction)")


if __name__ == "__main__":
    main()
    print("\n----- phase test -----")
    main2_phase_test()
