import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.decoder import Decoder

class Autoencoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        num_segments,
        n_freq_bins,
        channels,
        upsampling_factors,
        target_length,
        latent_mix_weight=0.0,
        decode_mix_weight=0.0,
        mrstft_weight=1.0,
        l1_weight=1.0,
        stft_loss_weight=0.0,
        mix_l1_weight=1.0,
        mix_mrstft_weight=1.0,
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
    ):
        super().__init__()
        self.eps = eps

        # MR-STFT params
        self.mrstft_ffts = mrstft_ffts
        self.mrstft_hops = mrstft_hops
        self.mrstft_wins = mrstft_wins
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight

        # STFT params for on-the-fly computation
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('stft_window', torch.hann_window(win_length))

        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            num_segments=num_segments,
            n_freq_bins=n_freq_bins,
            dropout=dropout,
        )

        self.decoder = Decoder(
            d_model=d_model,
            num_segments=num_segments,
            target_length=target_length,
            channels=channels,
            upsampling_factors=upsampling_factors,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            dropout=dropout,
        )

        self.l1_weight = l1_weight
        self.mrstft_weight = mrstft_weight
        self.stft_loss_weight = stft_loss_weight
        self.latent_mix_weight = latent_mix_weight
        self.decode_mix_weight = decode_mix_weight
        self.mix_l1_weight = mix_l1_weight
        self.mix_mrstft_weight = mix_mrstft_weight

    # -------------------------
    # MR-STFT helpers
    # -------------------------
    def _mrstft_mag(self, x, n_fft, hop, win):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.float()

        window = torch.hann_window(win, device=x.device, dtype=x.dtype)
        X = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, center=True, return_complex=True,
        )
        mag = torch.sqrt(X.real * X.real + X.imag * X.imag + self.eps)
        return mag

    def mrstft_loss(self, x_hat, x):
        total_sc = 0.0
        total_mag = 0.0

        for n_fft, hop, win in zip(self.mrstft_ffts, self.mrstft_hops, self.mrstft_wins):
            M = self._mrstft_mag(x, n_fft, hop, win)
            Mhat = self._mrstft_mag(x_hat, n_fft, hop, win)

            num = torch.linalg.norm(M - Mhat, dim=(1, 2))
            den = torch.linalg.norm(M, dim=(1, 2)).clamp(min=self.eps)
            sc = (num / den).mean()

            mag = F.l1_loss(torch.log(Mhat + self.eps), torch.log(M + self.eps))

            total_sc += sc
            total_mag += mag

        total_sc /= len(self.mrstft_ffts)
        total_mag /= len(self.mrstft_ffts)
        return self.sc_weight * total_sc + self.mag_weight * total_mag

    def _compute_stft_from_wave(self, x_wave):
        """Compute complex STFT from waveform on-the-fly (center=True)."""
        if x_wave.dim() == 3:
            x_wave = x_wave.squeeze(1)
        x_wave = x_wave.float()

        X = torch.stft(
            x_wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window,
            center=True,
            return_complex=True,
        )
        return torch.stack([X.real, X.imag], dim=1)

    # -------------------------
    # Mixing losses
    # -------------------------
    def latent_linearity_loss(self, x1_stft, x2_stft, z1, z2):
        B = x1_stft.size(0)
        alpha = torch.rand(B, 1, 1, 1, device=x1_stft.device)
        betta = 1 - alpha

        x_mix = alpha * x1_stft + betta * x2_stft
        z_mix = self.encoder(x_mix)

        alpha_z = alpha.squeeze(-1)
        betta_z = betta.squeeze(-1)
        z_mix_lin = alpha_z * z1 + betta_z * z2

        return F.mse_loss(z_mix, z_mix_lin)

    def decode_mixing_loss(self, z1, z2, x1_wave, x2_wave):
        B = z1.size(0)
        tgt = self.decoder.target_length
        if x1_wave.size(-1) > tgt:
            x1_wave = x1_wave[..., :tgt]
            x2_wave = x2_wave[..., :tgt]

        alpha = torch.rand(B, 1, 1, device=z1.device)
        beta = 1 - alpha

        x_mix_wave = alpha * x1_wave + beta * x2_wave

        # Interpolation path: mix in latent space and decode — this is the training signal
        z_interp = alpha * z1 + beta * z2
        x_interp = self.decoder(z_interp)

        l1_interp = F.l1_loss(x_interp, x_mix_wave)
        mr_interp = self.mrstft_loss(x_interp, x_mix_wave)
        loss_interp = self.mix_l1_weight * l1_interp + self.mix_mrstft_weight * mr_interp

        # Oracle path: encode the true mix and decode — no gradients needed
        # (only used for computing MixRate/Gap metrics, not for training)
        with torch.no_grad():
            x_mix_stft = self._compute_stft_from_wave(x_mix_wave)
            z_real = self.encoder(x_mix_stft)
            x_real_recon = self.decoder(z_real)
            l1_real = F.l1_loss(x_real_recon, x_mix_wave).item()
            mr_real = self.mrstft_loss(x_real_recon, x_mix_wave).item()
            loss_real_val = self.mix_l1_weight * l1_real + self.mix_mrstft_weight * mr_real
            latent_mix_error = F.mse_loss(z_real, z_interp).item()

        loss_interp_val = loss_interp.detach().item()
        rate = loss_interp_val / (loss_real_val + self.eps)
        gap = loss_interp_val - loss_real_val

        return {
            'total': loss_interp,
            'l1': l1_interp,
            'mrstft': mr_interp,
            'loss_real_val': loss_real_val,
            'l1_real': l1_real,
            'mr_real': mr_real,
            'rate': rate,
            'gap': gap,
            'latent_mix_error': latent_mix_error,
        }

    # -------------------------
    # Stem pair mixing
    # -------------------------
    def compute_stem_mixing_loss(self, x1_stft, x1_wave, x2_stft, x2_wave):
        if x1_wave.dim() == 2:
            x1_wave = x1_wave.unsqueeze(1)
        if x2_wave.dim() == 2:
            x2_wave = x2_wave.unsqueeze(1)

        z1 = self.encoder(x1_stft)
        z2 = self.encoder(x2_stft)
        return self.decode_mixing_loss(z1, z2, x1_wave, x2_wave)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x_stft, x_wave):
        """
        x_stft: [B, 2, F, T] — complex STFT (center=True)
        x_wave: [B, 1, L] — RMS-normalized waveform
        returns: total_loss, components
        """
        B = x_stft.size(0)
        assert x_wave.dim() == 3 and x_wave.size(1) == 1, "x_wave must be [B,1,L]"

        # Truncate x_wave to decoder target_length (handles padded preprocessing)
        tgt = self.decoder.target_length
        if x_wave.size(2) > tgt:
            x_wave = x_wave[:, :, :tgt]

        # Encode
        z = self.encoder(x_stft)

        # Decode with optional STFT prediction
        if self.stft_loss_weight > 0:
            x_hat, stft_pred = self.decoder(z, return_stft=True)
            # Direct STFT loss: predicted vs ground truth (both center=True domain)
            T_min = min(stft_pred.shape[-1], x_stft.shape[-1])
            stft_loss = F.l1_loss(stft_pred[..., :T_min], x_stft[..., :T_min])
        else:
            x_hat = self.decoder(z)
            stft_loss = torch.tensor(0.0, device=x_stft.device)

        # Waveform losses
        wav_l1 = F.l1_loss(x_hat, x_wave)
        mr = self.mrstft_loss(x_hat, x_wave)

        recon = (self.l1_weight * wav_l1
                 + self.mrstft_weight * mr
                 + self.stft_loss_weight * stft_loss)

        # Mixing losses
        latent_mix = torch.tensor(0.0, device=x_stft.device)
        decode_mix_total = torch.tensor(0.0, device=x_stft.device)
        decode_mix_l1 = torch.tensor(0.0, device=x_stft.device)
        decode_mix_mrstft = torch.tensor(0.0, device=x_stft.device)
        decode_mix_real = torch.tensor(0.0, device=x_stft.device)
        decode_mix_rate = 0.0
        decode_mix_gap = 0.0
        decode_mix_dict = {}

        if (self.latent_mix_weight > 0.0) or (self.decode_mix_weight > 0.0):
            assert B % 2 == 0, "Batch must be even when mixing is enabled"
            x1_stft, x2_stft = x_stft[:B // 2], x_stft[B // 2:]
            x1_wave, x2_wave = x_wave[:B // 2], x_wave[B // 2:]
            z1, z2 = z[:B // 2], z[B // 2:]

            if self.latent_mix_weight > 0.0:
                latent_mix = self.latent_linearity_loss(x1_stft, x2_stft, z1, z2)

            if self.decode_mix_weight > 0.0:
                decode_mix_dict = self.decode_mixing_loss(z1, z2, x1_wave, x2_wave)
                decode_mix_total = decode_mix_dict['total']
                decode_mix_l1 = decode_mix_dict['l1']
                decode_mix_mrstft = decode_mix_dict['mrstft']
                decode_mix_rate = decode_mix_dict['rate']
                decode_mix_gap = decode_mix_dict['gap']

        total = recon + self.latent_mix_weight * latent_mix + self.decode_mix_weight * decode_mix_total

        # Metrics
        rate = 0.0
        gap = 0.0
        decode_mix_real_val = 0.0
        decode_mix_real_l1 = 0.0
        decode_mix_real_mrstft = 0.0
        latent_mix_error = 0.0

        if self.decode_mix_weight > 0.0:
            rate = decode_mix_rate
            gap = decode_mix_gap
            decode_mix_real_val = decode_mix_dict.get('loss_real_val', 0.0)
            decode_mix_real_l1 = decode_mix_dict.get('l1_real', 0.0)
            decode_mix_real_mrstft = decode_mix_dict.get('mr_real', 0.0)
            latent_mix_error = decode_mix_dict.get('latent_mix_error', 0.0)

        components = {
            "total": total.detach().item(),
            "Latent/mean": z.mean().item(),
            "Latent/std": z.std().item(),
            "Latent/absmax": z.abs().max().item(),
            "ReconSingle/Total": recon.detach().item(),
            "ReconSingle/WavL1": wav_l1.detach().item(),
            "ReconSingle/MRSTFT": mr.detach().item(),
            "ReconSingle/STFTLoss": stft_loss.detach().item(),
            "MixReconInterp/Total": decode_mix_total.detach().item(),
            "MixReconInterp/WavL1": decode_mix_l1.detach().item(),
            "MixReconInterp/MRSTFT": decode_mix_mrstft.detach().item(),
            "MixReconReal/Total": decode_mix_real_val,
            "MixReconReal/WavL1": decode_mix_real_l1,
            "MixReconReal/MRSTFT": decode_mix_real_mrstft,
            "MixRate": rate,
            "MixGap": gap,
            "LatentMixError": latent_mix_error,
            "latent_mix": latent_mix.detach().item(),
        }
        return total, components
