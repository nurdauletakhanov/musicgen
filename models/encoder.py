"""1D waveform encoder (DAC-style strided convolutions + optional transformer)."""

import math
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.checkpoint import checkpoint

LEAKY_RELU_SLOPE = 0.1


class EncoderResBlock1d(nn.Module):
    """Residual block with dilated 1D convolutions.

    For each dilation d in dilations:
        h = LeakyReLU(x)
        h = Conv1d(ch, ch, k=3, dilation=d, pad=d) + weight_norm
        h = LeakyReLU(h)
        h = Conv1d(ch, ch, k=1) + weight_norm
        x = x + h
    """

    def __init__(self, channels, dilations=(1, 3, 9)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=3,
                                      dilation=d, padding=d)),
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=1)),
            ))

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class WaveEncoder(nn.Module):
    """1D convolutional encoder for raw waveforms.

    Compresses waveform [B, 1, L] to latent tokens [B, num_segments, d_model]
    using strided 1D convolutions followed by dilated residual blocks.

    Args:
        d_model: Latent dimension per token (e.g. 128).
        num_segments: Number of latent tokens (e.g. 22).
        encoder_strides: List of strides per stage (e.g. [6, 8, 8, 8]).
        encoder_channels: List of channels [initial, stage0, stage1, ...].
            Length = len(encoder_strides) + 1.
        encoder_dilations: Dilations for ResBlocks (e.g. [1, 3, 9]).
        encoder_kernel_scale: Kernel size = stride * scale (default 2).
        n_heads: Transformer attention heads (only if n_layers > 0).
        n_layers: Number of transformer layers (0 = pure conv).
        dropout: Dropout rate for transformer.
    """

    def __init__(self, d_model, num_segments,
                 encoder_strides=(6, 8, 8, 8),
                 encoder_channels=(32, 64, 128, 256, 256),
                 encoder_dilations=(1, 3, 9),
                 encoder_kernel_scale=2,
                 n_heads=4, n_layers=0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        self.encoder_strides = list(encoder_strides)
        self.total_stride = reduce(mul, encoder_strides, 1)
        self.padded_length = num_segments * self.total_stride

        n_stages = len(encoder_strides)
        assert len(encoder_channels) == n_stages + 1, \
            f"encoder_channels must have {n_stages + 1} elements, got {len(encoder_channels)}"

        # Initial conv (no stride)
        layers = [weight_norm(nn.Conv1d(1, encoder_channels[0],
                                        kernel_size=7, padding=3))]

        # Strided conv stages with residual blocks
        for i, stride in enumerate(encoder_strides):
            in_ch = encoder_channels[i]
            out_ch = encoder_channels[i + 1]
            kernel = stride * encoder_kernel_scale
            # DAC-style padding: ceil((kernel - stride) / 2). Matches pad = stride // 2
            # for even strides (v16's [4,8,8,4]) and gives the extra +1 needed to keep
            # output length exactly L / stride for odd strides (v1's [4,5,7,7]).
            pad = (kernel - stride + 1) // 2

            layers.append(nn.LeakyReLU(LEAKY_RELU_SLOPE))
            layers.append(weight_norm(nn.Conv1d(
                in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad)))
            layers.append(EncoderResBlock1d(out_ch, dilations=encoder_dilations))

        # Project to d_model
        layers.append(nn.LeakyReLU(LEAKY_RELU_SLOPE))
        layers.append(weight_norm(nn.Conv1d(encoder_channels[-1], d_model,
                                            kernel_size=1)))

        self.conv_stack = nn.ModuleList(layers)

        # Optional transformer
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

    def forward(self, x):
        """
        Args:
            x: [B, 1, L] waveform

        Returns:
            z: [B, num_segments, d_model] latent tokens
        """
        # Ensure mono channel
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Pad or truncate to padded_length
        L = x.size(2)
        if L < self.padded_length:
            x = F.pad(x, (0, self.padded_length - L))
        elif L > self.padded_length:
            x = x[:, :, :self.padded_length]

        # Conv stack
        h = x
        for layer in self.conv_stack:
            if self.training and isinstance(layer, EncoderResBlock1d):
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)

        # h: [B, d_model, num_segments]
        assert h.shape[2] == self.num_segments, \
            f"Expected {self.num_segments} tokens, got {h.shape[2]}"

        z = h.permute(0, 2, 1)  # [B, num_segments, d_model]

        # Optional transformer
        if self.use_transformer:
            z = z + self.pos_embed[:, :z.shape[1], :]
            for layer in self.transformer_layers:
                if self.training:
                    z = checkpoint(layer, z, use_reentrant=False)
                else:
                    z = layer(z)

        return z

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
