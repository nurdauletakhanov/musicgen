import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Shared architectural constants
LEAKY_RELU_SLOPE = 0.2
NUM_GROUPS = 8
NUM_RES_BLOCKS = 2


class ResBlock2D(nn.Module):
    """Residual block with Conv2D and GroupNorm for (freq, time) feature maps."""
    def __init__(self, channels, num_groups=NUM_GROUPS, kernel_size=(3, 3), dilation=(1, 1),
                 res_scale=0.1, dropout=0.1):
        super().__init__()
        self.res_scale = res_scale
        pad_f = dilation[0] * ((kernel_size[0] - 1) // 2)
        pad_t = dilation[1] * ((kernel_size[1] - 1) // 2)

        self.dropout = nn.Dropout2d(dropout)

        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                               padding=(pad_f, pad_t), dilation=dilation)

        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                               padding=(pad_f, pad_t), dilation=dilation)

        self.activation = nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=False)

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return x + self.res_scale * h


class Upsample2DBlock(nn.Module):
    """2D upsample: interpolate to target size, conv, then residual refinement."""
    def __init__(self, in_channels, out_channels, target_size,
                 kernel_size=(5, 3), num_res_blocks=NUM_RES_BLOCKS,
                 num_groups=NUM_GROUPS, dropout=0.1):
        super().__init__()
        self.target_size = target_size

        pad_f = (kernel_size[0] - 1) // 2
        pad_t = (kernel_size[1] - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=(pad_f, pad_t))
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=False)

        self.res_blocks = nn.Sequential(*[
            ResBlock2D(out_channels, num_groups=num_groups,
                       kernel_size=(3, 3), dropout=dropout)
            for _ in range(num_res_blocks)
        ])

    def forward(self, x):
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        x = self.activation(self.norm(self.conv(x)))
        x = self.res_blocks(x)
        return x


class Decoder(nn.Module):
    """Full-map 2D convolutional decoder for complex STFT prediction via iSTFT.

    Takes latent tokens [B, S, D], reshapes to a 2D spatial map [B, D, 1, S],
    and progressively upsamples in both frequency and time to reconstruct the
    full STFT [B, 2, F, T].

    Target sizes for each stage are derived automatically from the encoder's
    freq_strides and time_strides, ensuring the decoder exactly mirrors the
    encoder's spatial schedule in reverse.

    Number of upsample stages = len(freq_strides) + 1:
      - len(freq_strides) stages reverse the encoder's intermediate dims
      - 1 final stage reaches the full STFT resolution (n_freq_bins, n_stft_frames)

    Args:
        channels: List of N ints for decoder stage output channels, where
                  N = len(freq_strides) + 1. If None, defaults to 5-stage v4 layout.
        freq_strides: Encoder freq strides (used to derive decoder targets).
        time_strides: Encoder time strides (used to derive decoder targets).
    """

    def __init__(self, d_model, num_segments, target_length,
                 n_fft=1024, hop_length=256, win_length=1024,
                 dropout=0.0, num_refine_blocks=3,
                 channels=None, freq_strides=None, time_strides=None):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        self.target_length = target_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freq_bins = n_fft // 2 + 1  # 513 for n_fft=1024

        # --- Stride configuration (must match encoder) ---
        if freq_strides is not None:
            self.freq_strides = list(freq_strides)
        else:
            self.freq_strides = [2, 2, 2, 2]  # v4 default

        if time_strides is not None:
            self.time_strides = list(time_strides)
        else:
            self.time_strides = [1, 2, 3, 3]  # v4 default

        n_enc_blocks = len(self.freq_strides)
        assert len(self.time_strides) == n_enc_blocks

        # Compute total time stride and STFT frame count
        total_time_stride = 1
        for ts in self.time_strides:
            total_time_stride *= ts
        n_stft_frames = num_segments * total_time_stride

        # --- Compute encoder intermediate spatial dims ---
        # Replay the encoder's conv stack to get freq/time at each stage
        freq = self.n_freq_bins
        enc_freq_dims = []
        for i, fs in enumerate(self.freq_strides):
            k_f = 7 if i == 0 else 5
            p_f = 3 if i == 0 else 2
            freq = (freq + 2 * p_f - k_f) // fs + 1
            enc_freq_dims.append(freq)

        time = n_stft_frames
        enc_time_dims = []
        for ts in self.time_strides:
            time = (time + 2 * 1 - 3) // ts + 1  # k=3, p=1 for all time dims
            enc_time_dims.append(time)

        # Decoder targets = encoder dims reversed + full STFT resolution
        # E.g., v5 encoder: [171,57,19,7,3] → decoder: [3,7,19,57,171,513]
        freq_targets = list(reversed(enc_freq_dims)) + [self.n_freq_bins]
        time_targets = list(reversed(enc_time_dims)) + [n_stft_frames]

        n_stages = len(freq_targets)  # n_enc_blocks + 1

        # --- Decoder channel progression ---
        if channels is not None:
            assert len(channels) == n_stages, \
                f"channels must have {n_stages} elements (len(freq_strides)+1), got {len(channels)}"
            dec_channels = [d_model] + list(channels)
        else:
            # v4 default: 5 stages
            assert n_stages == 5, \
                f"Default channels only supports 5 stages (4 encoder blocks), got {n_stages}. Provide channels explicitly."
            dec_channels = [d_model, 256, 192, 128, 64, 32]

        # --- Build upsample blocks ---
        self.upsample_blocks = nn.ModuleList()
        for i in range(n_stages):
            ft = freq_targets[i]
            tt = time_targets[i]
            in_ch = dec_channels[i]
            out_ch = dec_channels[i + 1]
            freq_up = (i == 0) or (ft > freq_targets[i - 1])
            time_up = (tt > (num_segments if i == 0 else time_targets[i - 1]))
            kernel_map = {
                (True,  True):  (5, 3),
                (True,  False): (5, 1),
                (False, True):  (1, 3),
                (False, False): (3, 3),
            }
            k = kernel_map[(freq_up, time_up)]
            self.upsample_blocks.append(
                Upsample2DBlock(in_ch, out_ch, target_size=(ft, tt),
                                kernel_size=k, num_res_blocks=NUM_RES_BLOCKS,
                                dropout=dropout)
            )

        # Output refinement: dilated ResBlocks on full STFT map
        # Cyclic dilation schedule avoids exceeding the time dimension
        last_ch = dec_channels[-1]
        base_dilations = [1, 3, 9, 27, 81, 243]
        dilations = [(1, base_dilations[i % len(base_dilations)]) for i in range(num_refine_blocks)]
        self.out_refine = nn.Sequential(*[
            ResBlock2D(last_ch, kernel_size=(3, 3), dilation=d, dropout=dropout)
            for d in dilations
        ])
        # Vocos-style output: magnitude + unit-circle phase (3 channels)
        self.out_conv = nn.Conv2d(last_ch, 3, kernel_size=(3, 3), padding=(1, 1))

        # iSTFT window (registered buffer → moves with .to(device))
        self.register_buffer('istft_window', torch.hann_window(win_length))

    def forward(self, z):
        """
        Args:
            z: [B, S, D] latent tokens

        Returns:
            y: [B, 1, target_length] reconstructed waveform
            stft_pred: [B, 2, F, T] predicted STFT (real/imag)
        """
        B, S, D = z.shape
        assert S == self.num_segments, f"Expected {self.num_segments} segments, got {S}"

        # Reshape to 2D spatial map: [B, D, 1, S]
        x = z.permute(0, 2, 1).unsqueeze(2)  # [B, D, 1, S]

        # Progressive upsampling on the full map
        for block in self.upsample_blocks:
            if self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Output refinement
        if self.training:
            x = checkpoint(self.out_refine, x, use_reentrant=False)
        else:
            x = self.out_refine(x)

        out = self.out_conv(x)  # [B, 3, F, T]

        # Vocos-style: magnitude + unit-circle phase → valid complex STFT
        mag = F.softplus(out[:, 0])                        # [B, F, T] always positive
        cos_phi = out[:, 1]                                # [B, F, T]
        sin_phi = out[:, 2]                                # [B, F, T]
        # Normalize to unit circle (guarantees valid phase)
        norm = torch.sqrt(cos_phi ** 2 + sin_phi ** 2 + 1e-8)
        cos_phi = cos_phi / norm
        sin_phi = sin_phi / norm

        # Build real/imag for iSTFT and return as stft_pred
        real = mag * cos_phi
        imag = mag * sin_phi
        stft_pred = torch.stack([real, imag], dim=1)       # [B, 2, F, T]

        # iSTFT: cuFFT requires float32
        X = torch.complex(real.float(), imag.float())      # [B, F, T]

        wav = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.istft_window,
            center=True,
            length=self.target_length,
        )  # [B, target_length]

        wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        y = wav.unsqueeze(1)  # [B, 1, target_length]

        assert y.shape == (B, 1, self.target_length), \
            f"Expected (B, 1, {self.target_length}), got {y.shape}"
        return y, stft_pred

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
