"""Waveform-domain hybrid encoder (v15).

Takes a raw waveform [B, 1, L] and produces latent tokens [B, S, D] via a
DAC-style 1D-CNN trunk followed by a transformer over the token dimension.

Why waveform-in: STFT-input encoders (v6..v14) never generalized phase
through the 24x bottleneck. The dilated decoder reached +20 dB / 6 deg on
overfit with wave_l1, but full-data training stalled at Phase=90 deg
regardless of loss weighting. The discriminator finds a magnitude shortcut
and wave_l1 is a barrier (not a smooth gradient) between RMS-normalized
random-phase candidates. Every working continuous music AE (DAC, EnCodec,
Stable Audio VAE) encodes waveform-to-latent directly. v15 follows suit.

Architecture (defaults — 22050 Hz, 3s chunks, 22 tokens, d_model=128):

    [B, 1, 67584]                              (padded from 66150)
      -> Conv1d(1 -> 32, k=7)                  # initial projection
      -> EncoderBlock(32 -> 64,  stride=6)     # 2x ResidualUnit + stride conv
      -> EncoderBlock(64 -> 96,  stride=8)
      -> EncoderBlock(96 -> 128, stride=8)
      -> EncoderBlock(128 -> 128,stride=8)
    [B, 128, 22]
      -> permute to [B, 22, 128]
      -> +learned pos embed
      -> 6x TransformerEncoderLayer              (pre-norm, GELU)
    [B, 22, 128]

Downsample product = 6*8*8*8 = 3072. Input padded to 22*3072 = 67584.
Latent size: 22 * 128 = 2816 floats for 66150 samples -> 23.5x compression.

ResidualUnit follows DAC: dilated depthwise-style conv (k=7) -> GELU ->
pointwise conv (k=1) -> residual. Each block has 2 units with dilations
(1, 3) before a strided downsampling conv with kernel=2*stride, padding=
stride//2 (gives exact floor(L/stride) output when L is a multiple of
stride).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUnit(nn.Module):
    """DAC-style residual unit: dilated k=7 conv -> GELU -> k=1 conv + residual."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=7,
            padding=3 * dilation, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        h = F.gelu(self.conv1(x))
        h = self.conv2(h)
        return x + h


class EncoderBlock(nn.Module):
    """Two ResidualUnits then a strided downsampling conv."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.units = nn.Sequential(
            ResidualUnit(in_channels, dilation=1),
            ResidualUnit(in_channels, dilation=3),
        )
        self.downsample = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
        )

    def forward(self, x):
        x = self.units(x)
        x = F.gelu(self.downsample(x))
        return x


class Encoder(nn.Module):
    """Waveform-in hybrid CNN + transformer encoder.

    Args:
        d_model: latent token dim (also the last CNN channel count).
        n_heads: transformer heads.
        n_layers: transformer layers (0 disables the transformer).
        num_segments: expected number of output tokens (S).
        encoder_strides: downsampling strides per CNN block. Product is the
            total downsample factor; input is padded to num_segments * product.
        encoder_channels: per-block output channel counts. Length must equal
            len(encoder_strides); last entry should equal d_model.
        initial_channels: channels after the stem conv (before first block).
        dropout: transformer dropout.
    """

    def __init__(self, d_model, n_heads, n_layers, num_segments,
                 encoder_strides=(6, 8, 8, 8),
                 encoder_channels=(64, 96, 128, 128),
                 initial_channels=32,
                 dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments

        assert len(encoder_strides) == len(encoder_channels), \
            f"encoder_strides ({len(encoder_strides)}) and encoder_channels " \
            f"({len(encoder_channels)}) must match"
        assert encoder_channels[-1] == d_model, \
            f"last encoder_channels ({encoder_channels[-1]}) must equal d_model ({d_model})"

        self.encoder_strides = tuple(encoder_strides)
        self.encoder_channels = tuple(encoder_channels)
        self.total_stride = 1
        for s in encoder_strides:
            self.total_stride *= s
        self.required_L = num_segments * self.total_stride

        # Stem
        self.stem = nn.Conv1d(1, initial_channels, kernel_size=7, padding=3)

        # Strided CNN trunk
        blocks = []
        in_ch = initial_channels
        for out_ch, stride in zip(encoder_channels, encoder_strides):
            blocks.append(EncoderBlock(in_ch, out_ch, stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Transformer over tokens
        self.use_transformer = n_layers > 0
        if self.use_transformer:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_segments, d_model) * 0.02
            )
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    activation='gelu',
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ])

    def forward(self, x_wave):
        """
        Args:
            x_wave: [B, 1, L] waveform (any L <= required_L; gets right-padded).
        Returns:
            z: [B, num_segments, d_model] latent tokens
        """
        from torch.utils.checkpoint import checkpoint

        if x_wave.dim() == 2:
            x_wave = x_wave.unsqueeze(1)
        x_wave = x_wave.float()

        L = x_wave.size(-1)
        if L < self.required_L:
            x_wave = F.pad(x_wave, (0, self.required_L - L))
        elif L > self.required_L:
            x_wave = x_wave[..., :self.required_L]

        h = F.gelu(self.stem(x_wave))                       # [B, C0, L]
        h = self.blocks(h)                                   # [B, d_model, S]

        assert h.size(-1) == self.num_segments, \
            f"Encoder output length {h.size(-1)} != num_segments {self.num_segments}"
        assert h.size(1) == self.d_model, \
            f"Encoder output channels {h.size(1)} != d_model {self.d_model}"

        z = h.transpose(1, 2).contiguous()                   # [B, S, D]

        if self.use_transformer:
            z = z + self.pos_embed[:, :z.size(1), :]
            for layer in self.transformer_layers:
                if self.training:
                    z = checkpoint(layer, z, use_reentrant=False)
                else:
                    z = layer(z)

        return z

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
