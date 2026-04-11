import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.decoder import Decoder


class Autoencoder(nn.Module):
    """
    STFT autoencoder with:
      - Reconstruction: mel + MR-STFT magnitude
      - Optional decode-mixing loss (Option α: random cross-batch pairs)
      - Optional latent L2 regularization

    Mixing path (when decode_mix_weight > 0 and B >= 2):
      Pick a random non-trivial permutation π of batch positions.
      For each i, form (x_i, x_{π(i)}), pick α ~ U[0,1],
      mix in latent:    z_interp = α·z_i + (1-α)·z_{π(i)}
      mix in waveform:  x_mix    = α·x_i + (1-α)·x_{π(i)}
      Loss = L1(decoder(z_interp), x_mix)

    MixRate metric (validation only, gated by compute_mix_rate=True):
      oracle_l1 = L1(decoder(encoder(STFT(x_mix))), x_mix)
      rate = decode_mix_l1 / oracle_l1
      Values near 1.0 = latent interpolation is as good as re-encoding the real mix,
      i.e. the decoder is nearly linear in z.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        num_segments,
        n_freq_bins,
        target_length,
        decode_mix_weight=0.0,
        mrstft_weight=1.0,
        mel_weight=0.0,
        latent_l2_weight=0.0,
        dropout=0.1,
        mrstft_ffts=(1024, 2048, 512),
        mrstft_hops=(256, 512, 128),
        mrstft_wins=(1024, 2048, 512),
        sc_weight=1.0,
        mag_weight=1.0,
        eps=1e-8,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        num_refine_blocks=1,
        sample_rate=44100,
        channels=None,
        encoder_channels=None,
        freq_strides=None,
        time_strides=None,
    ):
        super().__init__()
        self.eps = eps

        # Loss weights
        self.mrstft_weight = mrstft_weight
        self.mel_weight = mel_weight
        self.decode_mix_weight = decode_mix_weight
        self.latent_l2_weight = latent_l2_weight

        # MR-STFT params
        self.mrstft_ffts = mrstft_ffts
        self.mrstft_hops = mrstft_hops
        self.mrstft_wins = mrstft_wins
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight

        for i, win_len in enumerate(mrstft_wins):
            self.register_buffer(f'_mrstft_win_{i}', torch.hann_window(win_len))

        # STFT params (used by mix-rate oracle)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.register_buffer('stft_window', torch.hann_window(win_length))

        # Mel filterbank (HTK formula, no librosa)
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

        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            num_segments=num_segments,
            n_freq_bins=n_freq_bins,
            dropout=dropout,
            encoder_channels=encoder_channels,
            freq_strides=freq_strides,
            time_strides=time_strides,
        )

        self.decoder = Decoder(
            d_model=d_model,
            num_segments=num_segments,
            target_length=target_length,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            dropout=dropout,
            num_refine_blocks=num_refine_blocks,
            channels=channels,
            freq_strides=freq_strides,
            time_strides=time_strides,
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

    def _compute_stft_from_wave(self, x_wave):
        """Compute complex STFT [B, 2, F, T] from waveform [B, 1, L]."""
        if x_wave.dim() == 3:
            x_wave = x_wave.squeeze(1)
        x_wave = x_wave.float()
        X = torch.stft(
            x_wave, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.stft_window,
            center=True, return_complex=True,
        )
        return torch.stack([X.real, X.imag], dim=1)

    # -------------------------
    # Mixing helper
    # -------------------------
    @staticmethod
    def _random_pair_perm(B, device):
        """Return a random permutation of [0, B) with no fixed points (no self-pairs)."""
        perm = torch.randperm(B, device=device)
        arange = torch.arange(B, device=device)
        # Retry up to 5 times to avoid fixed points; fall back to cyclic shift.
        for _ in range(5):
            if not (perm == arange).any():
                return perm
            perm = torch.randperm(B, device=device)
        return torch.roll(arange, shifts=max(1, B // 2))

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x_stft, x_wave, compute_mix_rate: bool = False):
        """
        Args:
            x_stft: [B, 2, F, T] — complex STFT (center=True)
            x_wave: [B, 1, L]   — RMS-normalized waveform
            compute_mix_rate: if True, compute MixRate oracle (validation only)

        Returns:
            total_loss, components dict, x_hat, stft_pred (only when not training)
        """
        B = x_stft.size(0)
        assert x_wave.dim() == 3 and x_wave.size(1) == 1, "x_wave must be [B,1,L]"

        tgt = self.decoder.target_length
        if x_wave.size(2) > tgt:
            x_wave = x_wave[:, :, :tgt]

        # Encode + decode
        z = self.encoder(x_stft)
        x_hat, stft_pred = self.decoder(z)

        # Reconstruction: mel + MR-STFT
        mr = self.mrstft_loss(x_hat, x_wave)
        mel = self.mel_loss(x_hat, x_wave) if self.mel_weight > 0.0 else x_hat.new_tensor(0.0)
        recon = self.mrstft_weight * mr + self.mel_weight * mel

        # Decode-mixing (Option α: random cross-batch pairs)
        decode_mix = x_hat.new_tensor(0.0)
        mix_rate = 0.0
        if self.decode_mix_weight > 0.0 and B >= 2:
            perm = self._random_pair_perm(B, x_stft.device)

            x2_wave = x_wave[perm]
            z2 = z[perm]

            alpha = torch.rand(B, 1, 1, device=x_stft.device)
            beta = 1.0 - alpha

            x_mix_wave = alpha * x_wave + beta * x2_wave
            z_interp = alpha * z + beta * z2

            x_interp, _ = self.decoder(z_interp)

            # Spectral (same scale as recon) + waveform L1 (phase-sensitive)
            mix_mr = self.mrstft_loss(x_interp, x_mix_wave)
            mix_mel = self.mel_loss(x_interp, x_mix_wave) if self.mel_weight > 0.0 else x_hat.new_tensor(0.0)
            mix_wav = F.l1_loss(x_interp, x_mix_wave)
            decode_mix = self.mrstft_weight * mix_mr + self.mel_weight * mix_mel + mix_wav

            if compute_mix_rate:
                with torch.no_grad():
                    x_mix_stft = self._compute_stft_from_wave(x_mix_wave)
                    z_real = self.encoder(x_mix_stft)
                    x_real_recon, _ = self.decoder(z_real)
                    l1_real = F.l1_loss(x_real_recon, x_mix_wave).item()
                    mix_rate = decode_mix.detach().item() / (l1_real + self.eps)

        # Latent L2 regularization
        if self.latent_l2_weight > 0.0:
            latent_l2 = self.latent_l2_weight * z.pow(2).mean()
        else:
            latent_l2 = x_hat.new_tensor(0.0)

        total = recon + latent_l2 + self.decode_mix_weight * decode_mix

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
            "MixRate": mix_rate,
        }
        return total, components, x_hat, (stft_pred if not self.training else None)
