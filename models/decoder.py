"""HiFi-GAN style 1D waveform decoder with Multi-Receptive-Field blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.checkpoint import checkpoint

LEAKY_RELU_SLOPE = 0.1


class DecoderResBlock1d(nn.Module):
    """Residual block for the HiFi-GAN decoder.

    For each dilation d in dilations:
        h = LeakyReLU(x)
        h = Conv1d(ch, ch, k=kernel_size, dilation=d) + weight_norm
        h = LeakyReLU(h)
        h = Conv1d(ch, ch, k=kernel_size, dilation=1) + weight_norm
        x = x + h
    """

    def __init__(self, channels, kernel_size=3, dilations=(1, 3)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            pad_d = d * (kernel_size - 1) // 2
            pad_1 = (kernel_size - 1) // 2
            self.blocks.append(nn.Sequential(
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                      dilation=d, padding=pad_d)),
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                      dilation=1, padding=pad_1)),
            ))

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class MRFBlock(nn.Module):
    """Multi-Receptive-Field Fusion block.

    Contains parallel DecoderResBlock1d with different kernel sizes.
    Outputs are averaged (HiFi-GAN V1 convention).
    """

    def __init__(self, channels, kernel_sizes=(3, 7), dilations=((1, 3), (1, 3))):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        self.resblocks = nn.ModuleList([
            DecoderResBlock1d(channels, kernel_size=k, dilations=d)
            for k, d in zip(kernel_sizes, dilations)
        ])

    def forward(self, x):
        return sum(rb(x) for rb in self.resblocks) / len(self.resblocks)


class WaveDecoder(nn.Module):
    """HiFi-GAN style 1D transposed-convolution decoder.

    Takes latent tokens [B, S, D], upsamples progressively with
    ConvTranspose1d + MRF blocks to reconstruct waveform [B, 1, L].

    Args:
        d_model: Latent dimension per token.
        num_segments: Number of latent tokens.
        target_length: Output waveform length (samples).
        decoder_channels: Channel sizes [initial_proj, stage0, stage1, ...].
            Length = len(decoder_strides) + 1.
        decoder_strides: Upsample strides per stage (reversed encoder strides).
        decoder_resblock_kernel_sizes: Kernel sizes for MRF ResBlocks.
        decoder_resblock_dilations: Dilations for each MRF ResBlock.
    """

    def __init__(self, d_model, num_segments, target_length,
                 decoder_channels=(256, 256, 128, 64, 32),
                 decoder_strides=(8, 8, 8, 6),
                 decoder_resblock_kernel_sizes=(3, 7),
                 decoder_resblock_dilations=((1, 3), (1, 3)),
                 encoder_kernel_scale=2):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        self.target_length = target_length

        n_stages = len(decoder_strides)
        assert len(decoder_channels) == n_stages + 1, \
            f"decoder_channels must have {n_stages + 1} elements, got {len(decoder_channels)}"

        # Input projection
        self.input_proj = weight_norm(nn.Conv1d(
            d_model, decoder_channels[0], kernel_size=7, padding=3))

        # Upsample stages
        self.up_blocks = nn.ModuleList()
        self.mrf_blocks = nn.ModuleList()

        for i, stride in enumerate(decoder_strides):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            kernel = stride * encoder_kernel_scale
            pad = stride * (encoder_kernel_scale - 1) // 2

            self.up_blocks.append(weight_norm(nn.ConvTranspose1d(
                in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad)))
            self.mrf_blocks.append(MRFBlock(
                out_ch,
                kernel_sizes=decoder_resblock_kernel_sizes,
                dilations=decoder_resblock_dilations,
            ))

        # Output projection
        self.output_conv = weight_norm(nn.Conv1d(
            decoder_channels[-1], 1, kernel_size=7, padding=3))

    def forward(self, z):
        """
        Args:
            z: [B, num_segments, d_model] latent tokens

        Returns:
            y: [B, 1, target_length] reconstructed waveform
            None: placeholder for backward compat (was stft_pred)
        """
        B, S, D = z.shape
        assert S == self.num_segments, f"Expected {self.num_segments} segments, got {S}"

        # [B, S, D] -> [B, D, S]
        x = z.permute(0, 2, 1)
        x = self.input_proj(x)

        # Progressive upsampling
        for up, mrf in zip(self.up_blocks, self.mrf_blocks):
            x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
            x = up(x)
            if self.training:
                x = checkpoint(mrf, x, use_reentrant=False)
            else:
                x = mrf(x)

        # Output
        x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
        x = self.output_conv(x)
        x = torch.tanh(x)

        # Crop to target length
        if x.size(2) > self.target_length:
            x = x[:, :, :self.target_length]
        elif x.size(2) < self.target_length:
            x = F.pad(x, (0, self.target_length - x.size(2)))

        return x, None

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
