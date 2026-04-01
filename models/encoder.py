import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import LEAKY_RELU_SLOPE, NUM_GROUPS


class ConvBlock2D(nn.Module):
    """Conv2D block with GroupNorm and LeakyReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 num_groups=NUM_GROUPS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        return F.leaky_relu(self.norm(self.conv(x)), LEAKY_RELU_SLOPE, inplace=True)


class Encoder(nn.Module):
    """
    Pure strided 2D CNN encoder for complex STFT spectrograms.

    Treats the STFT [B, 2, F, T] as a 2D image and compresses it to
    [B, num_segments, d_model] using only learned strided convolutions.

    Stride schedule is fully configurable via freq_strides and time_strides.
    After the strided conv stack, a learned freq_collapse Conv2D reduces the
    remaining frequency bins to 1, then an optional transformer mixes tokens.

    Example (v5, 5s @ 44.1kHz, n_fft=1024, hop=256):
      STFT:   [B, 2, 513, 864]
      Conv stack with freq_strides=(3,3,3,3,3), time_strides=(1,3,3,1,1):
        [B,48,171,864] → [B,96,57,288] → [B,96,19,96] → [B,96,7,96] → [B,96,3,96]
      freq_collapse (3,1): [B, 96, 1, 96]
      Latent: [B, 96, 96]
    """

    # Legacy class attributes for backward compat (decoder imports TOTAL_TIME_STRIDE)
    TIME_STRIDES = (1, 2, 3, 3)
    TOTAL_TIME_STRIDE = 18

    def __init__(self, d_model, n_heads, n_layers, num_segments,
                 n_freq_bins=513, dropout=0.0, encoder_channels=None,
                 freq_strides=None, time_strides=None):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        self.n_freq_bins = n_freq_bins

        # --- Stride configuration ---
        if freq_strides is not None:
            self.freq_strides = list(freq_strides)
        else:
            self.freq_strides = [2, 2, 2, 2]  # v4 default

        if time_strides is not None:
            self.time_strides = list(time_strides)
        else:
            self.time_strides = [1, 2, 3, 3]  # v4 default

        n_blocks = len(self.freq_strides)
        assert len(self.time_strides) == n_blocks, \
            f"freq_strides ({n_blocks}) and time_strides ({len(self.time_strides)}) must have same length"

        # Compute total time stride and required input T
        self.total_time_stride = 1
        for ts in self.time_strides:
            self.total_time_stride *= ts
        self._required_T = num_segments * self.total_time_stride

        # Compute freq dim after all strided convs
        # Each conv with kernel k, stride s, padding p: out = floor((in + 2p - k) / s) + 1
        # We use the same padding formula as the conv stack below
        freq = n_freq_bins
        self._freq_dims = [freq]  # track for debugging
        for i, fs in enumerate(self.freq_strides):
            k_f = 7 if i == 0 else 5
            p_f = 3 if i == 0 else 2
            freq = (freq + 2 * p_f - k_f) // fs + 1
            self._freq_dims.append(freq)
        self._freq_after_strides = freq

        # --- Channel configuration ---
        if encoder_channels is not None:
            assert len(encoder_channels) == n_blocks, \
                f"encoder_channels must have {n_blocks} elements, got {len(encoder_channels)}"
            ch = list(encoder_channels)
        else:
            ch = [32, 64, 128, 192]  # v4 default (only valid for 4 blocks)
            assert n_blocks == 4, \
                f"Default encoder_channels only supports 4 blocks, got {n_blocks}. Provide encoder_channels explicitly."

        # --- Conv stack ---
        blocks = []
        in_ch = 2
        for i in range(n_blocks):
            k_f = 7 if i == 0 else 5
            p_f = 3 if i == 0 else 2
            blocks.append(ConvBlock2D(
                in_ch, ch[i],
                kernel_size=(k_f, 3),
                stride=(self.freq_strides[i], self.time_strides[i]),
                padding=(p_f, 1),
            ))
            in_ch = ch[i]
        self.conv_stack = nn.Sequential(*blocks)

        # Learned frequency collapse: remaining freq bins → 1
        self.freq_collapse = nn.Conv2d(
            ch[-1], d_model,
            kernel_size=(self._freq_after_strides, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.freq_norm = nn.GroupNorm(NUM_GROUPS, d_model)

        # Optional transformer for cross-token context
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
            x: [B, 2, F, T] complex STFT (real/imag channels), float32
        Returns:
            z: [B, num_segments, d_model] latent tokens
        """
        from torch.utils.checkpoint import checkpoint

        B = x.shape[0]
        x = x.float()  # cuFFT / conv requires float32

        # Pad or truncate time dim to exactly required_T (= num_segments × total_time_stride)
        T = x.shape[-1]
        if T < self._required_T:
            x = F.pad(x, (0, self._required_T - T))
        elif T > self._required_T:
            x = x[..., :self._required_T]

        # Strided 2D conv stack
        h = self.conv_stack(x)
        assert h.shape[2] == self._freq_after_strides, \
            f"Freq dim after conv stack: expected {self._freq_after_strides}, got {h.shape[2]}"

        # Learned freq collapse
        h = self.freq_collapse(h)                          # [B, d_model, 1, num_segments]
        h = F.leaky_relu(self.freq_norm(h), LEAKY_RELU_SLOPE, inplace=True)

        # [B, d_model, 1, num_segments] → [B, num_segments, d_model]
        z = h.squeeze(2).permute(0, 2, 1)

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
