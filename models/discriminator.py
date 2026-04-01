"""Multi-Scale STFT Discriminator for adversarial audio training.

Based on the approach from EnCodec (Defossez et al., 2022) and DAC (Kumar et al., 2023).
Operates on STFT magnitudes at multiple FFT sizes to discriminate real vs generated audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


class STFTDiscriminator(nn.Module):
    """Single-scale STFT discriminator.

    Computes STFT magnitude of the input waveform, then applies a 2D conv
    network to produce patch-level real/fake predictions.
    """

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024,
                 channels=(32, 64, 128, 256, 256)):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.register_buffer('window', torch.hann_window(win_length))

        n_freq = n_fft // 2 + 1
        layers = []
        in_ch = 1  # STFT magnitude is single channel
        for out_ch in channels:
            layers.append(
                spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3),
                                        stride=(2, 2), padding=(1, 1)))
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        self.layers = nn.ModuleList()
        idx = 0
        for out_ch in channels:
            block = nn.Sequential(layers[idx], layers[idx + 1])
            self.layers.append(block)
            idx += 2

        self.output_conv = spectral_norm(
            nn.Conv2d(in_ch, 1, kernel_size=(3, 3), padding=(1, 1))
        )

    def _stft_mag(self, x):
        """Compute STFT magnitude from waveform."""
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.float()
        X = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            center=True, return_complex=True,
        )
        mag = torch.sqrt(X.real ** 2 + X.imag ** 2 + 1e-8)
        return mag.unsqueeze(1)  # [B, 1, F, T]

    def forward(self, x):
        """
        Args:
            x: [B, 1, L] waveform

        Returns:
            logits: patch-level real/fake predictions
            features: list of intermediate feature maps (for feature matching)
        """
        mag = self._stft_mag(x)
        features = []
        h = mag
        for layer in self.layers:
            h = layer(h)
            features.append(h)
        logits = self.output_conv(h)
        return logits, features


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-scale STFT discriminator.

    Applies STFTDiscriminator at multiple FFT sizes to capture different
    time-frequency resolutions.
    """

    def __init__(self, fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512),
                 win_sizes=(512, 1024, 2048),
                 channels=(32, 64, 128, 256, 256)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(
                n_fft=n_fft, hop_length=hop, win_length=win,
                channels=channels,
            )
            for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, 1, L] waveform

        Returns:
            all_logits: list of logit tensors (one per scale)
            all_features: list of feature lists (one per scale)
        """
        all_logits = []
        all_features = []
        for disc in self.discriminators:
            logits, features = disc(x)
            all_logits.append(logits)
            all_features.append(features)
        return all_logits, all_features

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class PeriodDiscriminator(nn.Module):
    """Single-period sub-discriminator from HiFi-GAN.

    Reshapes the waveform into a 2D grid by folding at period P, then applies
    2D convolutions to detect periodic artifacts (voiced sounds, guitar harmonics).
    Uses weight_norm (not spectral_norm) following the HiFi-GAN paper.
    """

    def __init__(self, period, channels=(32, 128, 512, 1024, 1024)):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList()
        in_ch = 1
        for out_ch in channels:
            self.layers.append(nn.Sequential(
                weight_norm(nn.Conv2d(in_ch, out_ch, (5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.1),
            ))
            in_ch = out_ch
        self.output_conv = weight_norm(nn.Conv2d(in_ch, 1, (3, 1), padding=(1, 0)))

    def forward(self, x):
        """
        Args:
            x: [B, 1, T] waveform

        Returns:
            logits: patch-level real/fake predictions
            features: list of intermediate feature maps (for feature matching)
        """
        x = x.squeeze(1)  # [B, T]
        B, T = x.shape
        pad = (self.period - T % self.period) % self.period
        if pad > 0:
            x = F.pad(x, (0, pad), mode='reflect')
        x = x.view(B, 1, -1, self.period)  # [B, 1, T//p, p]
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return self.output_conv(x), features


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD) from HiFi-GAN (Kong et al., 2020).

    Applies PeriodDiscriminator at prime periods [2,3,5,7,11] to detect
    periodic waveform artifacts characteristic of voiced/instrumental sounds.
    """

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x):
        all_logits, all_features = [], []
        for d in self.discriminators:
            logits, features = d(x)
            all_logits.append(logits)
            all_features.append(features)
        return all_logits, all_features

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class CombinedDiscriminator(nn.Module):
    """Combines MSSTFTD (spectral) + MPD (periodic) discriminators.

    Returns concatenated logits and features lists — fully compatible with
    the existing discriminator_loss / generator_loss / feature_matching_loss
    functions which iterate over lists without caring about the length.
    """

    def __init__(self, msstftd, mpd):
        super().__init__()
        self.msstftd = msstftd
        self.mpd = mpd

    def forward(self, x):
        stft_logits, stft_feats = self.msstftd(x)
        mpd_logits, mpd_feats = self.mpd(x)
        return stft_logits + mpd_logits, stft_feats + mpd_feats

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def discriminator_loss(real_logits, fake_logits):
    """Hinge loss for discriminator."""
    loss = 0.0
    for real_l, fake_l in zip(real_logits, fake_logits):
        loss += torch.mean(F.relu(1.0 - real_l)) + torch.mean(F.relu(1.0 + fake_l))
    return loss / len(real_logits)


def generator_loss(fake_logits):
    """Hinge loss for generator (adversarial)."""
    loss = 0.0
    for fake_l in fake_logits:
        loss += -torch.mean(fake_l)
    return loss / len(fake_logits)


def feature_matching_loss(real_features, fake_features):
    """L1 feature matching loss across all scales and layers."""
    loss = 0.0
    count = 0
    for real_feats, fake_feats in zip(real_features, fake_features):
        for real_f, fake_f in zip(real_feats, fake_feats):
            loss += F.l1_loss(fake_f, real_f.detach())
            count += 1
    return loss / max(count, 1)
