"""Multi-Scale STFT Discriminator for adversarial audio training.

DAC/EnCodec-style: operates on stacked [real, imag] STFT (not magnitude)
so the discriminator can see phase. Uses stride-(2,1) convs to downsample
frequency aggressively while preserving time resolution — this is the
architectural choice that lets the discriminator localize transients and
provide phase-sensitive gradients to the generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class STFTDiscriminator(nn.Module):
    """Single-FFT-size STFT discriminator (DAC-style).

    Input:  [B, 1, L] waveform
    STFT:   [B, 2, F, T]  (real, imag)
    Output: patch-level logits + intermediate features for feature matching
    """

    def __init__(self, n_fft, hop_length, win_length, channels=32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

        c = channels
        # 5 conv blocks: stride (2,1) — downsample freq, preserve time.
        # Wide time kernel (3,9) gives a large temporal receptive field per
        # layer for transient detection. Channel count stays flat — DAC convention.
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(2, c, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(c, c, (3, 9), stride=(2, 1), padding=(1, 4))),
            weight_norm(nn.Conv2d(c, c, (3, 9), stride=(2, 1), padding=(1, 4))),
            weight_norm(nn.Conv2d(c, c, (3, 9), stride=(2, 1), padding=(1, 4))),
            weight_norm(nn.Conv2d(c, c, (3, 3), padding=(1, 1))),
        ])
        self.out_conv = weight_norm(nn.Conv2d(c, 1, (3, 3), padding=(1, 1)))

    def _stft(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.float()
        X = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            center=True, return_complex=True,
        )
        return torch.stack([X.real, X.imag], dim=1)  # [B, 2, F, T]

    def forward(self, x):
        h = self._stft(x)
        features = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            features.append(h)
        return self.out_conv(h), features


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-scale STFT discriminator over 3 FFT sizes — DAC default."""

    def __init__(self,
                 fft_sizes=(2048, 1024, 512),
                 hop_sizes=(512, 256, 128),
                 win_sizes=(2048, 1024, 512),
                 channels=32):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n_fft=n, hop_length=h, win_length=w, channels=channels)
            for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, x):
        all_logits, all_features = [], []
        for d in self.discriminators:
            logits, features = d(x)
            all_logits.append(logits)
            all_features.append(features)
        return all_logits, all_features

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
