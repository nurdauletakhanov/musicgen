import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from models.encoder import Encoder
from models.decoder import Decoder


class Autoencoder(nn.Module):
    """Waveform-in / waveform-out autoencoder (v15).

    Encoder: DAC-style strided 1D-CNN + transformer over tokens.
        [B, 1, L] -> [B, S, D]
    Decoder: ConvTranspose1d upsampling stack mirroring the encoder.
        [B, S, D] -> [B, 1, target_length]

    Loss recipe (DAC-aligned, no phase-targeted losses):
        total = lambda_mel       * log_mel_l1(x_hat, x_wave)
              + lambda_mrstft    * mrstft_mag_l1(x_hat, x_wave)
              + lambda_l2        * latent.pow(2).mean()
              (+ adversarial + feat-match added by trainer when disc is on)

    Direct losses (mel + MR-STFT mag) carry magnitude; the discriminator
    (run by the trainer) carries phase realism. Unlike v14, v15 drops the
    wave_l1 phase prior because the waveform-domain encoder/decoder pair
    propagates phase natively through the bottleneck.

    x_stft is kept in the forward signature for backward compat with the
    trainer and the mixing losses, but the encoder only consumes x_wave.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        num_segments,
        target_length,
        # Decode-mixing loss (random-pair linearity; off by default)
        decode_mix_weight=0.0,
        # Mix-recon loss (recon on the real mix α·x₁+β·x₂, teaches decoder
        # that mixes are in-distribution audio). Uses the same mel + MRSTFT
        # weights as single-stem recon. Off by default.
        mix_recon_weight=0.0,
        # v15 recon recipe (DAC-style)
        mel_l1_weight=15.0,
        mrstft_mag_l1_weight=1.0,
        latent_l2_weight=0.001,
        dropout=0.1,
        eps=1e-8,
        # STFT params (used only for auxiliary losses / mixing oracle)
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sample_rate=22050,
        # v15 encoder architecture
        encoder_strides=(6, 8, 8, 8),
        encoder_channels=(64, 96, 128, 128),
        encoder_initial_channels=32,
        # v15 decoder architecture
        decoder_strides=(8, 8, 8, 6),
        decoder_channels=(128, 96, 64, 32),
        # Ignored (kept for backward compat with old configs)
        n_freq_bins=None,
        freq_strides=None,
        time_strides=None,
        complex_l1_weight=0.0,
        wave_l1_weight=0.0,
        decoder_hidden_dim=None,
        decoder_dilations=None,
    ):
        super().__init__()
        self.eps = eps

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.register_buffer('stft_window', torch.hann_window(win_length))

        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            num_segments=num_segments,
            encoder_strides=encoder_strides,
            encoder_channels=encoder_channels,
            initial_channels=encoder_initial_channels,
            dropout=dropout,
        )

        self.decoder = Decoder(
            d_model=d_model,
            num_segments=num_segments,
            target_length=target_length,
            decoder_strides=decoder_strides,
            decoder_channels=decoder_channels,
        )

        self.mel_l1_weight = mel_l1_weight
        self.mrstft_mag_l1_weight = mrstft_mag_l1_weight
        self.latent_l2_weight = latent_l2_weight

        # MR-STFT magnitude L1 scales (DAC default).
        self.mrstft_scales = [
            (2048, 512),
            (1024, 256),
            (512, 128),
            (256, 64),
            (128, 32),
        ]
        for n_fft_s, _ in self.mrstft_scales:
            self.register_buffer(
                f'_mrstft_win_{n_fft_s}',
                torch.hann_window(n_fft_s),
                persistent=False,
            )

        # 80-band log-mel @ sample_rate (DAC/EnCodec/BigVGAN convention).
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=80,
            f_min=0.0,
            f_max=sample_rate / 2,
            power=1.0,
            center=True,
            normalized=False,
        )
        self.decode_mix_weight = decode_mix_weight
        self.mix_recon_weight = mix_recon_weight

    # -------------------------
    # Helpers
    # -------------------------
    def _mrstft_mag_l1(self, x_hat, x_wave):
        """Multi-resolution STFT magnitude L1 across DAC's 5 scales."""
        xh = x_hat.squeeze(1).float()
        xt = x_wave.squeeze(1).float()
        loss = 0.0
        for n_fft_s, hop_s in self.mrstft_scales:
            window = getattr(self, f'_mrstft_win_{n_fft_s}')
            Xh = torch.stft(
                xh, n_fft=n_fft_s, hop_length=hop_s, win_length=n_fft_s,
                window=window, center=True, return_complex=True,
            )
            Xt = torch.stft(
                xt, n_fft=n_fft_s, hop_length=hop_s, win_length=n_fft_s,
                window=window, center=True, return_complex=True,
            )
            loss = loss + F.l1_loss(Xh.abs(), Xt.abs())
        return loss / len(self.mrstft_scales)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x_stft, x_wave, compute_mix_rate: bool = False):
        """
        x_stft: kept for backward compat with the trainer; unused in v15.
        x_wave: [B, 1, L] waveform target.
        compute_mix_rate: if True, additionally runs the (no_grad) oracle —
            encode the real mix and decode it — to compute MixRate. Only
            enable during validation; it doubles encoder+decoder compute.
        returns: total_loss, components, x_hat, None
        """
        assert x_wave.dim() == 3 and x_wave.size(1) == 1, "x_wave must be [B,1,L]"
        B = x_wave.size(0)

        # Truncate target waveform to decoder output length.
        tgt = self.decoder.target_length
        if x_wave.size(2) > tgt:
            x_wave = x_wave[:, :, :tgt]

        z = self.encoder(x_wave)
        x_hat = self.decoder(z)

        zero = x_wave.new_zeros(())
        mel_l1 = zero
        mrstft_mag = zero

        if self.mel_l1_weight > 0.0:
            mel_pred = self.mel_transform(x_hat.squeeze(1).float())
            mel_tgt = self.mel_transform(x_wave.squeeze(1).float())
            mel_l1 = F.l1_loss(torch.log(mel_pred + 1e-5),
                               torch.log(mel_tgt + 1e-5))
        if self.mrstft_mag_l1_weight > 0.0:
            mrstft_mag = self._mrstft_mag_l1(x_hat, x_wave)

        recon = (
            self.mel_l1_weight * mel_l1
            + self.mrstft_mag_l1_weight * mrstft_mag
        )

        # Mixing-path losses: run when either decode-mix or mix-recon is on.
        # Random-pair mixing: split the current batch in half, draw per-pair
        # alpha, and build x_mix = alpha*x1 + beta*x2. Both losses share the
        # same alpha/pair to keep them mutually consistent.
        decode_mix = zero
        mix_recon = zero
        mix_mel = zero
        mix_mrstft = zero
        mix_rate = 0.0

        mix_on = (self.decode_mix_weight > 0.0 or self.mix_recon_weight > 0.0)
        if mix_on and B >= 2:
            half = B // 2
            x1_wave = x_wave[:half]
            x2_wave = x_wave[half:2 * half]
            z1 = z[:half]
            z2 = z[half:2 * half]

            alpha = torch.rand(half, 1, 1, device=x_wave.device)
            beta = 1.0 - alpha
            x_mix_wave = alpha * x1_wave + beta * x2_wave

            # --- Decode-mix (linearity): decode linear-interp latent, compare
            # against the waveform mix. Supervises decoder linearity in z. ---
            if self.decode_mix_weight > 0.0:
                z_interp = alpha * z1 + beta * z2
                x_interp = self.decoder(z_interp)
                decode_mix = F.l1_loss(x_interp, x_mix_wave)

                if compute_mix_rate:
                    with torch.no_grad():
                        z_real = self.encoder(x_mix_wave)
                        x_real_recon = self.decoder(z_real)
                        l1_real = F.l1_loss(x_real_recon, x_mix_wave).item()
                    mix_rate = decode_mix.detach().item() / (l1_real + self.eps)

            # --- Mix-recon: encode+decode the real mix and apply the same
            # mel+MRSTFT recon recipe as single-stem recon. Teaches the
            # decoder that mixes are in-distribution audio. Same weights as
            # single-stem recon, scaled by mix_recon_weight as a single knob. ---
            if self.mix_recon_weight > 0.0:
                z_mix_real = self.encoder(x_mix_wave)
                x_mix_recon = self.decoder(z_mix_real)
                if self.mel_l1_weight > 0.0:
                    mix_mel = F.l1_loss(
                        torch.log(self.mel_transform(x_mix_recon.squeeze(1).float()) + 1e-5),
                        torch.log(self.mel_transform(x_mix_wave.squeeze(1).float()) + 1e-5),
                    )
                if self.mrstft_mag_l1_weight > 0.0:
                    mix_mrstft = self._mrstft_mag_l1(x_mix_recon, x_mix_wave)
                mix_recon = (
                    self.mel_l1_weight * mix_mel
                    + self.mrstft_mag_l1_weight * mix_mrstft
                )

        latent_l2 = self.latent_l2_weight * z.pow(2).mean() if self.latent_l2_weight > 0.0 else zero

        total = (
            recon
            + latent_l2
            + self.decode_mix_weight * decode_mix
            + self.mix_recon_weight * mix_recon
        )

        components = {
            "total": total.detach().item(),
            "Latent/mean": z.mean().item(),
            "Latent/std": z.std().item(),
            "Latent/absmax": z.abs().max().item(),
            "ReconSingle/Total": recon.detach().item(),
            "ReconSingle/MelL1": mel_l1.detach().item(),
            "ReconSingle/MRSTFTMag": mrstft_mag.detach().item(),
            "DecodeMix/L1": decode_mix.detach().item() if isinstance(decode_mix, torch.Tensor) else float(decode_mix),
            "MixRecon/Total": mix_recon.detach().item() if isinstance(mix_recon, torch.Tensor) else float(mix_recon),
            "MixRecon/MelL1": mix_mel.detach().item() if isinstance(mix_mel, torch.Tensor) else float(mix_mel),
            "MixRecon/MRSTFTMag": mix_mrstft.detach().item() if isinstance(mix_mrstft, torch.Tensor) else float(mix_mrstft),
            "Latent/l2": latent_l2.detach().item() if isinstance(latent_l2, torch.Tensor) else float(latent_l2),
        }
        if compute_mix_rate and self.decode_mix_weight > 0.0:
            components["MixRate"] = mix_rate
        return total, components, x_hat, None
