"""Wave-to-wave autoencoder with spectral reconstruction losses."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import WaveEncoder
from models.decoder import WaveDecoder


class Autoencoder(nn.Module):
    """
    Waveform autoencoder with:
      - Reconstruction: MR-STFT magnitude + optional mel
      - Optional decode-mixing loss (Option alpha: random cross-batch pairs)
      - Optional latent L2 regularization

    Encoder and decoder both operate in the waveform domain.
    Phase is implicit in the temporal structure -- no explicit STFT prediction.
    """

    def __init__(
        self,
        d_model,
        num_segments,
        target_length,
        # Encoder params
        encoder_strides=(6, 8, 8, 8),
        encoder_channels=(32, 64, 128, 256, 256),
        encoder_dilations=(1, 3, 9),
        encoder_kernel_scale=2,
        n_heads=4,
        n_layers=0,
        dropout=0.0,
        # Decoder params
        decoder_channels=(256, 256, 128, 64, 32),
        decoder_resblock_kernel_sizes=(3, 7),
        decoder_resblock_dilations=((1, 3), (1, 3)),
        # Loss params
        decode_mix_weight=0.0,
        latent_mix_weight=0.0,
        mrstft_weight=1.0,
        mel_weight=0.0,
        latent_l2_weight=0.0,
        mrstft_ffts=(1024, 2048, 512),
        mrstft_hops=(256, 512, 128),
        mrstft_wins=(1024, 2048, 512),
        sc_weight=1.0,
        mag_weight=1.0,
        eps=1e-8,
        sample_rate=22050,
        # Mel params (for loss computation only, not architectural)
        n_fft=1024,
        hop_length=256,
        win_length=1024,
    ):
        super().__init__()
        self.eps = eps
        self.sample_rate = sample_rate

        # Loss weights
        self.mrstft_weight = mrstft_weight
        self.mel_weight = mel_weight
        self.decode_mix_weight = decode_mix_weight
        self.latent_mix_weight = latent_mix_weight
        self.latent_l2_weight = latent_l2_weight

        # MR-STFT params
        self.mrstft_ffts = mrstft_ffts
        self.mrstft_hops = mrstft_hops
        self.mrstft_wins = mrstft_wins
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight

        for i, win_len in enumerate(mrstft_wins):
            self.register_buffer(f'_mrstft_win_{i}', torch.hann_window(win_len))

        # Mel filterbank (for loss computation)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        if mel_weight > 0.0:
            n_mels, fmin, fmax = 80, 20.0, sample_rate / 4.0
            mel_min = 2595 * np.log10(1 + fmin / 700)
            mel_max = 2595 * np.log10(1 + fmax / 700)
            mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
            hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
            bins = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
            fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
            for m in range(1, n_mels + 1):
                for k in range(bins[m - 1], bins[m]):
                    fb[m - 1, k] = (k - bins[m - 1]) / max(bins[m] - bins[m - 1], 1)
                for k in range(bins[m], bins[m + 1]):
                    fb[m - 1, k] = (bins[m + 1] - k) / max(bins[m + 1] - bins[m], 1)
            self.register_buffer('_mel_fb', torch.from_numpy(fb))
            self.register_buffer('_mel_win', torch.hann_window(n_fft))

        # STFT window for phase/consistency diagnostics (computed from waveforms)
        self.register_buffer('stft_window', torch.hann_window(win_length))

        self.encoder = WaveEncoder(
            d_model=d_model,
            num_segments=num_segments,
            encoder_strides=encoder_strides,
            encoder_channels=encoder_channels,
            encoder_dilations=encoder_dilations,
            encoder_kernel_scale=encoder_kernel_scale,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.decoder = WaveDecoder(
            d_model=d_model,
            num_segments=num_segments,
            target_length=target_length,
            decoder_channels=decoder_channels,
            decoder_strides=list(reversed(encoder_strides)),
            decoder_resblock_kernel_sizes=decoder_resblock_kernel_sizes,
            decoder_resblock_dilations=decoder_resblock_dilations,
            encoder_kernel_scale=encoder_kernel_scale,
        )

    # -------------------------
    # Spectral losses
    # -------------------------
    def _mrstft_mag(self, x, n_fft, hop, window):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.float()
        X = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=window.size(0),
            window=window, center=True, return_complex=True,
        )
        return torch.sqrt(X.real * X.real + X.imag * X.imag + self.eps)

    def mrstft_loss(self, x_hat, x):
        total_sc = 0.0
        total_mag = 0.0

        if x.dim() == 3:
            x = x.squeeze(1)
        x_f32 = x.detach().float()

        for i, (n_fft, hop) in enumerate(zip(self.mrstft_ffts, self.mrstft_hops)):
            window = getattr(self, f'_mrstft_win_{i}')

            with torch.no_grad():
                M = self._mrstft_mag(x_f32, n_fft, hop, window)
            Mhat = self._mrstft_mag(x_hat, n_fft, hop, window)

            num = torch.linalg.norm(M - Mhat, dim=(1, 2))
            den = torch.linalg.norm(M, dim=(1, 2)).clamp(min=self.eps)
            sc = (num / den).mean()
            mag = F.l1_loss(torch.log(Mhat + self.eps), torch.log(M + self.eps))

            total_sc += sc
            total_mag += mag

        total_sc /= len(self.mrstft_ffts)
        total_mag /= len(self.mrstft_ffts)
        return self.sc_weight * total_sc + self.mag_weight * total_mag

    def mel_loss(self, x_hat, x):
        """L1 on log-mel spectrogram."""
        def _log_mel(wav):
            if wav.dim() == 3:
                wav = wav.squeeze(1)
            wav = wav.float()
            X = torch.stft(
                wav, n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length, window=self._mel_win,
                center=True, return_complex=True,
            )
            mag = torch.sqrt(X.real ** 2 + X.imag ** 2 + self.eps)
            mel = torch.einsum('nf,bft->bnt', self._mel_fb, mag)
            return torch.log(mel.clamp(min=self.eps))

        with torch.no_grad():
            mel_real = _log_mel(x.detach())
        return F.l1_loss(_log_mel(x_hat), mel_real)

    # -------------------------
    # Encoder helpers (VAE / AE uniform API)
    # -------------------------
    def encode_mean(self, x_wave):
        """Return the deterministic latent (mu if VAE, else z).

        Used by mixing paths, stem-pair code, and anywhere we need the
        same latent that inference uses. Works for wave-mode or STFT-mode
        encoders and for variational or deterministic heads.
        """
        if x_wave.dim() == 2:
            x_wave = x_wave.unsqueeze(1)
        out = self.encoder(x_wave)
        if getattr(self, 'variational', False):
            mu, _ = out
            return mu
        return out

    # -------------------------
    # Mixing helper
    # -------------------------
    @staticmethod
    def _random_pair_perm(B, device):
        """Return a random permutation of [0, B) with no fixed points."""
        perm = torch.randperm(B, device=device)
        arange = torch.arange(B, device=device)
        for _ in range(5):
            if not (perm == arange).any():
                return perm
            perm = torch.randperm(B, device=device)
        return torch.roll(arange, shifts=max(1, B // 2))

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x_wave, compute_mix_rate: bool = False):
        """
        Args:
            x_wave: [B, 1, L] — waveform
            compute_mix_rate: if True, compute MixRate oracle (validation only)

        Returns:
            total_loss, components dict, x_hat, None
        """
        B = x_wave.size(0)
        if x_wave.dim() == 2:
            x_wave = x_wave.unsqueeze(1)

        tgt = self.decoder.target_length
        if x_wave.size(2) > tgt:
            x_wave = x_wave[:, :, :tgt]

        # Encode + decode
        z = self.encoder(x_wave)
        x_hat, _ = self.decoder(z)

        # Reconstruction: MR-STFT + optional mel
        mr = self.mrstft_loss(x_hat, x_wave)
        mel = self.mel_loss(x_hat, x_wave) if self.mel_weight > 0.0 else x_hat.new_tensor(0.0)
        recon = self.mrstft_weight * mr + self.mel_weight * mel

        # Mixing pairs (shared by decode-mix, latent-mix, and MixRate)
        decode_mix = x_hat.new_tensor(0.0)
        latent_mix = x_hat.new_tensor(0.0)
        mix_rate = 0.0
        mix_gap = 0.0
        # mix_aux carries the mixed-decode output and the matching real-mix
        # waveform out to the trainer when v2's `disc_on_mix` path needs them.
        mix_aux = None
        need_pairs = (self.decode_mix_weight > 0.0 or self.latent_mix_weight > 0.0
                      or compute_mix_rate) and B >= 2

        if need_pairs:
            perm = self._random_pair_perm(B, x_wave.device)
            alpha = torch.rand(B, 1, 1, device=x_wave.device)
            beta = 1.0 - alpha

            x_mix_wave = alpha * x_wave + beta * x_wave[perm]
            z_interp = alpha * z + beta * z[perm]

            # Decode-mixing loss
            if self.decode_mix_weight > 0.0 or compute_mix_rate:
                x_interp, _ = self.decoder(z_interp)
                mix_mr = self.mrstft_loss(x_interp, x_mix_wave)
                mix_mel = self.mel_loss(x_interp, x_mix_wave) if self.mel_weight > 0.0 else x_hat.new_tensor(0.0)
                mix_wav = F.l1_loss(x_interp, x_mix_wave)
                decode_mix = self.mrstft_weight * mix_mr + self.mel_weight * mix_mel + mix_wav
                mix_gap = decode_mix.detach().item()
                # Surface the mixed-decode output for the trainer's
                # disc_on_mix path. Only set when L_dec is active in the
                # generator's gradient graph (not when compute_mix_rate-only).
                if self.decode_mix_weight > 0.0:
                    mix_aux = {"x_interp": x_interp, "x_mix_wave": x_mix_wave}

            # MixRate (validation only)
            if compute_mix_rate:
                with torch.no_grad():
                    z_real = self.encoder(x_mix_wave)
                    x_real_recon, _ = self.decoder(z_real)
                    oracle_mr = self.mrstft_loss(x_real_recon, x_mix_wave)
                    oracle_mel = self.mel_loss(x_real_recon, x_mix_wave) if self.mel_weight > 0.0 else x_hat.new_tensor(0.0)
                    oracle_wav = F.l1_loss(x_real_recon, x_mix_wave)
                    oracle_loss = (self.mrstft_weight * oracle_mr + self.mel_weight * oracle_mel + oracle_wav).item()
                    mix_rate = mix_gap / (oracle_loss + self.eps)

            # Latent-mixing loss
            if self.latent_mix_weight > 0.0:
                z_mix_encoded = self.encoder(x_mix_wave)
                latent_mix = F.mse_loss(z_mix_encoded, z_interp)

        # Latent L2 regularization
        if self.latent_l2_weight > 0.0:
            latent_l2 = self.latent_l2_weight * z.pow(2).mean()
        else:
            latent_l2 = x_hat.new_tensor(0.0)

        total = (recon + latent_l2
                 + self.decode_mix_weight * decode_mix
                 + self.latent_mix_weight * latent_mix)

        components = {
            "total": total.detach().item(),
            "Latent/mean": z.mean().item(),
            "Latent/std": z.std().item(),
            "Latent/absmax": z.abs().max().item(),
            "Latent/l2": latent_l2.detach().item(),
            "ReconSingle/Total": recon.detach().item(),
            "ReconSingle/MRSTFT": mr.detach().item(),
            "ReconSingle/Mel": mel.detach().item(),
            "DecodeMix/L1": decode_mix.detach().item(),
            "LatentMix/MSE": latent_mix.detach().item(),
            "MixGap": mix_gap,
            "MixRate": mix_rate,
        }
        return total, components, x_hat, mix_aux
