"""Waveform-domain decoder (v15).

Mirrors the v15 encoder. Takes latent tokens [B, S, D] and upsamples to a
raw waveform [B, 1, L_target] via a stack of ConvTranspose1d upsampling
blocks, each preceded by ResidualUnits.

Architecture (defaults — 22 tokens, d_model=128, 3s @ 22050 Hz):

    [B, 22, 128] -> permute -> [B, 128, 22]
      -> Conv1d(128 -> 128, k=7)                   # initial projection
      -> DecoderBlock(128 -> 128, stride=8)        # ConvT + 2x ResidualUnit
      -> DecoderBlock(128 -> 96,  stride=8)
      -> DecoderBlock(96  -> 64,  stride=8)
      -> DecoderBlock(64  -> 32,  stride=6)
      -> Conv1d(32 -> 1, k=7) -> tanh
    [B, 1, 67584]                                    (cropped to target_length)

Upsample product = 8*8*8*6 = 3072. Output length = 22 * 3072 = 67584,
cropped to target_length (66150 for 3s @ 22050).

ConvTranspose1d uses kernel=2*stride, padding=stride//2, which gives
exact out = in*stride (matches the encoder's floor(L/s) downsampling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Legacy constants kept for any lingering imports.
LEAKY_RELU_SLOPE = 0.2
NUM_GROUPS = 8

from models.encoder import ResidualUnit


class DecoderBlock(nn.Module):
    """Strided transposed conv upsample then two ResidualUnits."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
        )
        self.units = nn.Sequential(
            ResidualUnit(out_channels, dilation=1),
            ResidualUnit(out_channels, dilation=3),
        )

    def forward(self, x):
        x = F.gelu(self.upsample(x))
        x = self.units(x)
        return x


class Decoder(nn.Module):
    """Waveform-out decoder mirroring the v15 encoder.

    Args:
        d_model: latent token dim (first CNN channel count).
        num_segments: number of input tokens (S in [B, S, D]).
        target_length: output waveform length in samples. CNN natively
            produces num_segments * prod(decoder_strides); output is cropped.
        decoder_strides: upsampling strides per block.
        decoder_channels: per-block OUTPUT channel counts. Length must equal
            len(decoder_strides).
        final_channels: input channels for the final 1x1 conv to mono.
    """

    def __init__(self, d_model, num_segments, target_length,
                 decoder_strides=(8, 8, 8, 6),
                 decoder_channels=(128, 96, 64, 32)):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        self.target_length = target_length

        assert len(decoder_strides) == len(decoder_channels), \
            f"decoder_strides ({len(decoder_strides)}) and decoder_channels " \
            f"({len(decoder_channels)}) must match"

        self.decoder_strides = tuple(decoder_strides)
        self.decoder_channels = tuple(decoder_channels)
        self.total_stride = 1
        for s in decoder_strides:
            self.total_stride *= s
        self.cnn_out_length = num_segments * self.total_stride

        # Initial projection
        self.in_proj = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)

        blocks = []
        in_ch = d_model
        for out_ch, stride in zip(decoder_channels, decoder_strides):
            blocks.append(DecoderBlock(in_ch, out_ch, stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.out_conv = nn.Conv1d(in_ch, 1, kernel_size=7, padding=3)

    def forward(self, z):
        """
        Args:
            z: [B, S, D] latent tokens
        Returns:
            wave: [B, 1, target_length] waveform in [-1, 1]
        """
        B, S, D = z.shape
        assert S == self.num_segments, f"Expected S={self.num_segments}, got {S}"
        assert D == self.d_model, f"Expected D={self.d_model}, got {D}"

        x = z.transpose(1, 2).contiguous()                   # [B, D, S]
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_conv(x)
        x = torch.tanh(x)

        if x.size(-1) > self.target_length:
            x = x[..., :self.target_length]
        elif x.size(-1) < self.target_length:
            x = F.pad(x, (0, self.target_length - x.size(-1)))
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
