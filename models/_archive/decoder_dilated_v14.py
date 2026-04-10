"""Single dilated 1D-CNN STFT decoder (v14).

A clean DAC-aligned decoder. Maps [B, S, D] latent tokens to a full STFT
[B, 2, F, S*fpt] in one shot through a stack of residual dilated 1D convs
over the token dimension, then iSTFTs to a waveform.

Why this design:
  - No phase/magnitude split. Once phase is supervised by a discriminator
    (and not by an explicit complex L1) there's no reason to isolate the
    two — both benefit from cross-token context (formants, attack envelopes,
    harmonic phase coherence all span many frames).
  - No L2-normalization on the output. With the discriminator handling
    phase realism, the "predict unit direction" trick is solving a problem
    that no longer exists, and the L2-normalize op pinches gradient at
    small magnitudes.
  - No log/clamp/exp chain on magnitude. Predict raw (real, imag) STFT
    bins; let mel L1 + MR-STFT mag L1 + adversarial supervise the result.
  - Residual dilated convs: WaveNet-style block, well-understood, gives
    full-chunk receptive field with O(log S) depth.

Architecture (defaults):
    [B, S=29, D=96]
        -> permute to [B, D, S]
        -> Conv1d(D -> H=256, k=1)                 # input projection
        -> [residual block (k=3, dil=1)]   ╮
        -> [residual block (k=3, dil=2)]   │  4 dilated blocks,
        -> [residual block (k=3, dil=4)]   │  GroupNorm + GELU
        -> [residual block (k=3, dil=8)]   ╯
        -> Conv1d(H -> fpt*2*F, k=1)               # output expansion
        -> reshape to [B, 2, F, S*fpt]
        -> iSTFT -> [B, 1, target_length]

Receptive field over tokens with dilations (1,2,4,8) and k=3:
    1 + 2*(1+2+4+8) = 31 tokens — full chunk (S=29) in both directions.

Param count at H=256, S=29, fpt=9, F=513: ~3.2M total.

The residual blocks use post-norm: x + GELU(GroupNorm(Conv(x))). The output
conv uses default init (NOT zero-init): with magnitude-only losses the
gradient at exactly x_hat=0 is 0/0 -> 0, which leaves the optimizer
stranded on a saddle. Small random output is necessary for mel/MR-STFT
gradients to flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared constants — encoder still imports these
LEAKY_RELU_SLOPE = 0.2
NUM_GROUPS = 8
NUM_RES_BLOCKS = 2  # legacy, unused, kept to avoid import breakage


class _ResidualDilatedBlock(nn.Module):
    """k=3 dilated Conv1d + GroupNorm + GELU, residual."""

    def __init__(self, channels, dilation, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation,
        )
        self.norm = nn.GroupNorm(n_groups, channels)

    def forward(self, x):
        return x + F.gelu(self.norm(self.conv(x)))


class Decoder(nn.Module):
    """Dilated 1D-CNN STFT decoder + iSTFT.

    Args:
        d_model: latent token dim.
        num_segments: number of latent tokens (S in [B, S, D]).
        n_freq_bins: STFT frequency bins (n_fft//2 + 1).
        frames_per_token: number of STFT frames each token produces. Should
            equal the encoder's total time stride.
        target_length: output waveform length in samples.
        n_fft, hop_length, win_length: iSTFT params.
        hidden_dim: channel width inside the dilated CNN trunk.
        dilations: dilation schedule for the residual blocks.
        n_groups: GroupNorm groups inside each block.
    """

    def __init__(self, d_model, num_segments, n_freq_bins,
                 frames_per_token, target_length,
                 n_fft, hop_length, win_length,
                 hidden_dim=256, dilations=(1, 2, 4, 8), n_groups=8):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        self.n_freq_bins = n_freq_bins
        self.frames_per_token = frames_per_token
        self.target_length = target_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        H = hidden_dim
        self.in_proj = nn.Conv1d(d_model, H, kernel_size=1)

        self.blocks = nn.ModuleList([
            _ResidualDilatedBlock(H, dilation=d, n_groups=n_groups)
            for d in dilations
        ])

        out_channels = frames_per_token * 2 * n_freq_bins
        self.out_conv = nn.Conv1d(H, out_channels, kernel_size=1)
        # NOTE: do NOT zero-init the output here. With magnitude-only losses
        # (mel L1, MR-STFT mag L1) the gradient of |STFT(x_hat)| at x_hat=0
        # is 0/0 -> PyTorch returns zero -> the optimizer is stuck at a
        # zero-gradient saddle. Default Conv1d init produces small random
        # noise, which gives the magnitude losses something to bite on.

        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, z):
        """
        Args:
            z: [B, S, D] latent tokens
        Returns:
            wave: [B, 1, target_length] waveform
            stft_pred: [B, 2, F, S*fpt] predicted STFT (raw real/imag).
        """
        B, S, D = z.shape
        assert S == self.num_segments, f"Expected S={self.num_segments}, got {S}"
        assert D == self.d_model, f"Expected D={self.d_model}, got {D}"

        x = z.transpose(1, 2)                       # [B, D, S]
        x = self.in_proj(x)                         # [B, H, S]
        for block in self.blocks:
            x = block(x)                            # residual inside
        x = self.out_conv(x)                        # [B, fpt*2*F, S]

        # Reshape [B, fpt*2*F, S] -> [B, 2, F, S*fpt].
        # Group fpt and (2*F) inside the channel dim, then unfold.
        x = x.view(B, self.frames_per_token, 2, self.n_freq_bins, S)
        # [B, fpt, 2, F, S] -> [B, 2, F, S, fpt] -> [B, 2, F, S*fpt]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        stft_pred = x.view(B, 2, self.n_freq_bins, S * self.frames_per_token)

        complex_stft = torch.complex(stft_pred[:, 0].float(),
                                     stft_pred[:, 1].float())
        wave = torch.istft(
            complex_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=self.target_length,
        )
        return wave.unsqueeze(1), stft_pred

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
