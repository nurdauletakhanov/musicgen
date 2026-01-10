import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.decoder import Decoder  # your waveform decoder

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
        latent_mix_weight=0.0,      # weight for latent linearity (optional)
        decode_mix_weight=0.0,      # weight for waveform-space mix (recommended)
        mrstft_weight=1.0,
        l1_weight=1.0,
        mix_l1_weight=1.0,          # weight for L1 loss in decode mixing
        mix_mrstft_weight=1.0,      # weight for MRSTFT loss in decode mixing
        dropout=0.1,
        mrstft_ffts=(1024, 2048, 512),
        mrstft_hops=(256, 512, 128),
        mrstft_wins=(1024, 2048, 512),
        sc_weight=1.0,
        mag_weight=1.0,
        stft_center=False,
        eps=1e-8,
        # STFT params for on-the-fly computation (decode mixing)
        n_fft=1024,
        hop_length=256,
        win_length=1024,
    ):
        super().__init__()
        self.eps = eps
        self.stft_center = stft_center

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
        # Register Hann window as buffer (moves with model to GPU)
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
            channels=channels,
            upsampling_factors=upsampling_factors,
            target_length=target_length,
            dropout=dropout,
        )

        self.l1_weight = l1_weight
        self.mrstft_weight = mrstft_weight
        self.latent_mix_weight = latent_mix_weight
        self.decode_mix_weight = decode_mix_weight
        self.mix_l1_weight = mix_l1_weight
        self.mix_mrstft_weight = mix_mrstft_weight

    # -------------------------
    # MR-STFT helpers
    # -------------------------
    def _mrstft_mag(self, x, n_fft, hop, win):
        """
        x: [B, 1, L] or [B, L]
        returns: [B, F, T] magnitude
        """
        if x.dim() == 3:
            x = x.squeeze(1)  # [B, L]
        x = x.float()  # STFT in fp32 for stability

        window = torch.hann_window(win, device=x.device, dtype=x.dtype)
        X = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=self.stft_center,
            return_complex=True,
        )
        mag = torch.sqrt(X.real * X.real + X.imag * X.imag + self.eps)
        return mag

    def mrstft_loss(self, x_hat, x):
        total_sc = 0.0
        total_mag = 0.0

        for n_fft, hop, win in zip(self.mrstft_ffts, self.mrstft_hops, self.mrstft_wins):
            M = self._mrstft_mag(x, n_fft, hop, win)
            Mhat = self._mrstft_mag(x_hat, n_fft, hop, win)

            # Spectral convergence per-batch (use Fro norm)
            num = torch.linalg.norm(M - Mhat, dim=(1, 2))
            den = torch.linalg.norm(M, dim=(1, 2)).clamp(min=self.eps)
            sc = (num / den).mean()

            # Log-mag L1
            mag = F.l1_loss(torch.log(Mhat + self.eps), torch.log(M + self.eps))

            total_sc += sc
            total_mag += mag

        total_sc /= len(self.mrstft_ffts)
        total_mag /= len(self.mrstft_ffts)
        return self.sc_weight * total_sc + self.mag_weight * total_mag

    def _compute_stft_from_wave(self, x_wave):
        """
        Compute complex STFT from waveform on-the-fly.
        
        Args:
            x_wave: [B, 1, L] waveform tensor
            
        Returns:
            stft: [B, 2, F, T] STFT with real/imag channels
        """
        if x_wave.dim() == 3:
            x_wave = x_wave.squeeze(1)  # [B, L]
        x_wave = x_wave.float()  # STFT in fp32 for stability
        
        X = torch.stft(
            x_wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window,
            center=False,  # Same as preprocessing
            return_complex=True,
        )
        # X: [B, F, T] complex -> [B, 2, F, T] real/imag
        return torch.stack([X.real, X.imag], dim=1)

    # -------------------------
    # Mixing losses
    # -------------------------
    def latent_linearity_loss(self, x1_stft, x2_stft, z1, z2):
        """
        Enforce: E(alpha*x1 + betta*x2) ≈ alpha*E(x1) + betta*E(x2)
        x*_stft: [B, 2, F, T]
        """
        B = x1_stft.size(0)
        alpha = torch.rand(B, 1, 1, 1, device=x1_stft.device)  # for STFT mixing
        betta = 1 - alpha  # for STFT mixing


        x_mix = (alpha * x1_stft + betta * x2_stft)

        z_mix = self.encoder(x_mix)


        alpha_z = alpha.squeeze(-1)  # [B,1,1] broadcasts over [S,D]
        betta_z = betta.squeeze(-1)  # [B,1,1] broadcasts over [S,D]

        z_mix_lin = (alpha_z * z1 + betta_z * z2) 

        return F.mse_loss(z_mix, z_mix_lin)

    def decode_mixing_loss(self, z1, z2, x1_wave, x2_wave):
        """
        Compare latent interpolation vs real autoencoder on mixed input.
        
        Both losses computed against same target (mixed waveform) for fair comparison:
        - MixReconReal: D(E(mixed_stft)) vs x_mix_wave
        - MixReconInterp: D(alpha*z1 + beta*z2) vs x_mix_wave
        - MixRate: MixReconInterp / MixReconReal (ideal = 1.0)
        - MixGap: MixReconInterp - MixReconReal
        - LatentMixError: || E(x_mix) - (alpha*z1 + beta*z2) ||^2 (no grad)
        
        z*: [B, S, D]
        x*_wave: [B, 1, L]
        Returns dict with loss components and metrics
        """
        B = z1.size(0)
        alpha = torch.rand(B, 1, 1, device=z1.device)     # for z/wave mixing
        beta = 1 - alpha
        
        # Mix waveforms (target for both losses)
        # alpha is [B, 1, 1], x_wave is [B, 1, L] - broadcasts correctly
        x_mix_wave = alpha * x1_wave + beta * x2_wave
        
        # Compute STFT on-the-fly and run through full autoencoder
        x_mix_stft = self._compute_stft_from_wave(x_mix_wave)
        z_real = self.encoder(x_mix_stft)
        x_real_recon = self.decoder(z_real)
        
        # Decode interpolated latent
        z_interp = alpha * z1 + beta * z2
        x_interp = self.decoder(z_interp)
        
        # MixReconReal: Loss for real autoencoder on mixed input
        l1_real = F.l1_loss(x_real_recon, x_mix_wave)
        mr_real = self.mrstft_loss(x_real_recon, x_mix_wave)
        loss_real = self.mix_l1_weight * l1_real + self.mix_mrstft_weight * mr_real
        
        # MixReconInterp: Loss for latent interpolation
        l1_interp = F.l1_loss(x_interp, x_mix_wave)
        mr_interp = self.mrstft_loss(x_interp, x_mix_wave)
        loss_interp = self.mix_l1_weight * l1_interp + self.mix_mrstft_weight * mr_interp
        
        # MixRate and MixGap (key metrics)
        loss_real_val = loss_real.detach().item()
        loss_interp_val = loss_interp.detach().item()
        rate = loss_interp_val / (loss_real_val + self.eps)
        gap = loss_interp_val - loss_real_val
        
        # LatentMixError: || E(x_mix) - (alpha*z1 + beta*z2) ||^2 (no gradient)
        with torch.no_grad():
            latent_mix_error = F.mse_loss(z_real, z_interp).item()
        
        return {
            'total': loss_interp,  # Use interp loss for training
            # MixReconInterp breakdown
            'l1': l1_interp,
            'mrstft': mr_interp,
            # MixReconReal breakdown
            'loss_real': loss_real,
            'l1_real': l1_real.detach().item(),
            'mr_real': mr_real.detach().item(),
            # Key metrics
            'rate': rate,
            'gap': gap,
            'latent_mix_error': latent_mix_error,
        }

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x_stft, x_wave):
        """
        x_stft: [B, 2, F, T]
        x_wave: [B, 1, L]  (RMS-normalized)
        returns: total_loss, components
        """
        B = x_stft.size(0)
        assert x_wave.dim() == 3 and x_wave.size(1) == 1, "x_wave must be [B,1,L]"

        # Recon
        z = self.encoder(x_stft)          # [B, S, D]
        x_hat = self.decoder(z)           # [B, 1, L]

        wav_l1 = F.l1_loss(x_hat, x_wave)
        mr = self.mrstft_loss(x_hat, x_wave)

        recon = self.l1_weight * wav_l1 + self.mrstft_weight * mr

        # Mixing
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
            x1_stft, x2_stft = x_stft[: B // 2], x_stft[B // 2 :]
            x1_wave, x2_wave = x_wave[: B // 2], x_wave[B // 2 :]
            z1, z2 = z[: B // 2], z[B // 2 :]

            if self.latent_mix_weight > 0.0:
                latent_mix = self.latent_linearity_loss(x1_stft, x2_stft, z1, z2)

            if self.decode_mix_weight > 0.0:
                decode_mix_dict = self.decode_mixing_loss(z1, z2, x1_wave, x2_wave)
                decode_mix_total = decode_mix_dict['total']
                decode_mix_l1 = decode_mix_dict['l1']
                decode_mix_mrstft = decode_mix_dict['mrstft']
                decode_mix_real = decode_mix_dict['loss_real']
                decode_mix_rate = decode_mix_dict['rate']
                decode_mix_gap = decode_mix_dict['gap']

        total = recon + self.latent_mix_weight * latent_mix + self.decode_mix_weight * decode_mix_total
        
        # Initialize metric values
        rate = 0.0
        gap = 0.0
        decode_mix_real_val = 0.0
        decode_mix_real_l1 = 0.0
        decode_mix_real_mrstft = 0.0
        latent_mix_error = 0.0
        
        if self.decode_mix_weight > 0.0:
            rate = decode_mix_rate
            gap = decode_mix_gap
            decode_mix_real_val = decode_mix_real.detach().item()
            decode_mix_real_l1 = decode_mix_dict.get('l1_real', 0.0)
            decode_mix_real_mrstft = decode_mix_dict.get('mr_real', 0.0)
            latent_mix_error = decode_mix_dict.get('latent_mix_error', 0.0)

        components = {
            # Total loss
            "total": total.detach().item(),
            # ReconSingle
            "ReconSingle/Total": recon.detach().item(),
            "ReconSingle/WavL1": wav_l1.detach().item(),
            "ReconSingle/MRSTFT": mr.detach().item(),
            # MixReconInterp
            "MixReconInterp/Total": decode_mix_total.detach().item(),
            "MixReconInterp/WavL1": decode_mix_l1.detach().item(),
            "MixReconInterp/MRSTFT": decode_mix_mrstft.detach().item(),
            # MixReconReal
            "MixReconReal/Total": decode_mix_real_val,
            "MixReconReal/WavL1": decode_mix_real_l1,
            "MixReconReal/MRSTFT": decode_mix_real_mrstft,
            # Key metrics
            "MixRate": rate,
            "MixGap": gap,
            "LatentMixError": latent_mix_error,
            # Legacy names (for backward compatibility with existing logs)
            "latent_mix": latent_mix.detach().item(),
        }
        return total, components
