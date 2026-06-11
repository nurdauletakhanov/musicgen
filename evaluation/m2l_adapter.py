"""Adapter that exposes a music2latent (M2L) checkpoint behind the v2
``Autoencoder`` interface used by ``compute_mixing_metrics`` and
``compute_fad``.

The M2L architecture is a 2-D STFT encoder/decoder + consistency UNet, very
different from this repo's wave-domain GAN autoencoder. The eval scripts only
need a small surface — ``model.encoder(wave) -> z``,
``model.decoder(z) -> (wave, _)``, ``model(wave) -> (_, _, wave_hat, _)``,
and the MR-STFT/Mel buffers — so we wrap M2L behind that surface here.

A 1 s @ 44.1 kHz clip (44100 samples) crops to 42496 samples for M2L
(``((((L - 3*hop) // hop) // ds) * hop * ds) + 3*hop`` with hop=512, ds=8),
giving 10 latent tokens. That cropped length is what we expose as
``decoder.target_length``; the eval slices its 44100-sample chunks down to it
before encoding, so M2L and v2 see audio of comparable length but not bit-for-bit
identical (M2L gives up the trailing ~36 ms per chunk).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from music2latent import hparams as hp
from music2latent.audio import to_representation_encoder, to_waveform
from music2latent.inference import EncoderDecoder
from music2latent.models import UNet


# ---------------------------------------------------------------------------
# Inner wrappers that present the M2L UNet's encode/decode as standalone
# callables matching the v2 ``model.encoder`` / ``model.decoder`` interface.

class _M2LEncoderWrapper(nn.Module):
    """Wraps ``UNet.encoder``: STFT-encoder front-end takes waveform in, latent out.

    Reflection-pads the input waveform to ``internal_length`` (a length the
    M2L crop rule maps to itself). The matching decoder slices the
    decoded waveform back to ``target_length``. Together this gives a
    chunk-continuous full-track reconstruction (no clicks at chunk
    boundaries when ``compute_fad`` concatenates per-chunk decodes), and
    aligns with the v2 chunk size (44100 samples) so the M2L row of the
    headline table is directly comparable to the v2 rows.
    """

    def __init__(self, gen: UNet, internal_length: int):
        super().__init__()
        self.gen = gen
        self.internal_length = internal_length

    def forward(self, x_wave: torch.Tensor) -> torch.Tensor:
        # x_wave: [B, 1, L] or [B, L]
        if x_wave.dim() == 3:
            x_wave = x_wave.squeeze(1)
        x_wave = x_wave.float()
        # Reflection-pad up to the M2L-natural internal length.
        cur_L = x_wave.size(-1)
        if cur_L < self.internal_length:
            pad = self.internal_length - cur_L
            x_wave = torch.nn.functional.pad(x_wave, (0, pad), mode='reflect')
        elif cur_L > self.internal_length:
            x_wave = x_wave[:, :self.internal_length]
        repr_enc = to_representation_encoder(x_wave)  # [B, 2, hop*2, T_stft]
        return self.gen.encoder(repr_enc)             # [B, bottleneck_channels, T_lat]


class _M2LDecoderWrapper(nn.Module):
    """Wraps the deterministic 1-step decode: latent -> STFT -> waveform.

    The noise that seeds ``forward_generator`` is derived from a fixed seed
    per call (``noise_seed``, default 0). The consistency-model decoder picks
    a *random* time-domain phase for every decode, so SI-SDR between two
    decodes of the same latent is essentially −∞ unless they share the
    seeding noise. By seeding deterministically we make ``sdr_lin =
    SI-SDR(g(z̄), g(f(x̄)))`` meaningful: both calls start from the same
    initial noise, so any difference in output is attributable to the
    latent difference (which is exactly what mixing-equivariance measures),
    not to phase randomness in the diffusion init.

    ``sdr_rec = SI-SDR(g(f(x)), x)`` is still phase-incoherent with ``x``
    and remains uninformative for any consistency-model decoder — that's
    a fact about the model class, not the adapter.
    """

    def __init__(self, gen: UNet, target_length: int, noise_seed: int = 0):
        super().__init__()
        self.gen = gen
        self.target_length = target_length
        self.noise_seed = noise_seed

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, None]:
        B = z.size(0)
        downscaling = 2 ** hp.freq_downsample_list.count(0)
        T_stft = int(z.shape[-1] * downscaling)
        # Deterministic noise: every decode call with the same shape and
        # noise_seed produces the same initial_noise. Within one batch the
        # three calls (g(f(x)), g(z̄), g(f(x̄))) all share the same noise.
        gen_noise = torch.Generator(device=z.device).manual_seed(self.noise_seed)
        initial_noise = torch.randn(
            (B, hp.data_channels, hp.hop * 2, T_stft),
            device=z.device, dtype=torch.float32, generator=gen_noise,
        ) * hp.sigma_max
        if z.dtype != torch.float32:
            initial_noise = initial_noise.to(z.dtype)
        pyramid = self.gen.decoder(z)
        x_stft = self.gen(z, initial_noise, sigma=hp.sigma_max, pyramid_latents=pyramid)
        x_wave = to_waveform(x_stft)
        x_wave = x_wave[:, :self.target_length]
        return x_wave.unsqueeze(1), None


# ---------------------------------------------------------------------------
# The main adapter

class M2LAutoencoderAdapter(nn.Module):
    """Presents an M2L checkpoint behind the v2 Autoencoder eval surface.

    The MR-STFT / Mel state is copied verbatim from
    ``models/autoencoder.py`` so that
    ``evaluation/compute_mixing_metrics._per_sample_recon_loss`` works
    against this adapter without modification.
    """

    def __init__(
        self,
        m2l_checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        # MR-STFT / Mel params: defaults match v2.0_continued.yaml so the
        # MixRate denominator/numerator scale matches v2 rows.
        mrstft_ffts=(256, 512, 1024, 2048),
        mrstft_hops=(64, 128, 256, 512),
        mrstft_wins=(256, 512, 1024, 2048),
        sc_weight: float = 1.0,
        mag_weight: float = 1.0,
        mel_weight: float = 1.0,
        eps: float = 1e-8,
        sample_rate: int = 44100,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
    ):
        super().__init__()
        # Load the M2L checkpoint via EncoderDecoder. EncoderDecoder
        # downloads music2latent.pt automatically if missing and loads
        # gen_state_dict (the published checkpoint already stores
        # EMA-merged weights here).
        ed = EncoderDecoder(load_path_inference=m2l_checkpoint_path, device=device)
        self.gen: UNet = ed.gen
        self.gen.eval()
        for p in self.gen.parameters():
            p.requires_grad = False

        # Internal compute length: smallest M2L-natural length >= sample_rate.
        # The crop rule maps L -> ((((L-3h)//h)//ds)*h*ds) + 3h. For 44100,
        # the natural cropped length is 42496 (loses 1604 samples). Padding
        # input up to the next M2L-natural length (46592) and slicing the
        # decoded output back to sample_rate (44100) gives chunk-continuous
        # full-track reconstruction (no clicks at chunk boundaries) and
        # exact-44100 outputs that align with the v2 chunk size.
        downscaling = 2 ** hp.freq_downsample_list.count(0)
        L = sample_rate
        natural_at_L = ((((L - 3 * hp.hop) // hp.hop) // downscaling) * hp.hop * downscaling) + 3 * hp.hop
        if natural_at_L >= L:
            internal_length = natural_at_L
        else:
            # next-larger M2L-natural length
            n = ((L - 3 * hp.hop) // hp.hop) // downscaling + 1
            internal_length = n * hp.hop * downscaling + 3 * hp.hop
        # For 44100 / hop=512 / ds=8: internal_length = 46592, target = 44100.
        target_length = sample_rate

        self.encoder = _M2LEncoderWrapper(self.gen, internal_length=internal_length)
        self.decoder = _M2LDecoderWrapper(self.gen, target_length=target_length)

        # ---- v2 Autoencoder MR-STFT / Mel state (verbatim from
        # models/autoencoder.py:59-100, with mel_weight==0.0 always
        # registering the buffers — the v2 code only registers if mel_weight>0,
        # but the eval path always touches them when mel_weight>0 in eval).
        self.eps = eps
        self.sample_rate = sample_rate
        self.mrstft_ffts = mrstft_ffts
        self.mrstft_hops = mrstft_hops
        self.mrstft_wins = mrstft_wins
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight
        self.mel_weight = mel_weight

        for i, win_len in enumerate(mrstft_wins):
            self.register_buffer(f'_mrstft_win_{i}', torch.hann_window(win_len))

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

    # The v2 Autoencoder methods mrstft_loss/mel_loss/_mrstft_mag are also
    # consumed by some eval paths. Re-implement them inline below so the
    # adapter is self-contained.

    def _mrstft_mag(self, x, n_fft, hop, window):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.float()
        X = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=window.size(0),
            window=window, center=True, return_complex=True,
        )
        return torch.sqrt(X.real * X.real + X.imag * X.imag + self.eps)

    @torch.no_grad()
    def forward(self, x_wave: torch.Tensor):
        """v2 Autoencoder.forward returns (loss_dict, comps, x_hat, mix_aux).

        The eval scripts only consume index [2] (x_hat). Return None for the
        other slots.
        """
        if x_wave.dim() == 2:
            x_wave = x_wave.unsqueeze(1)
        # Slice to target_length (matches what compute_mixing_metrics does
        # explicitly, and also what compute_fad's _reconstruct_full_track
        # implicitly relies on via x_ref = x[:, :, :tgt]).
        x_wave = x_wave[:, :, :self.decoder.target_length]
        z = self.encoder(x_wave)
        x_hat, _ = self.decoder(z)
        return None, None, x_hat, None
