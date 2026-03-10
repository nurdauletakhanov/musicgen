import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ResBlock2D(nn.Module):
    """Residual block with Conv2D and GroupNorm for (freq, time) feature maps."""
    def __init__(self, channels, num_groups=8, kernel_size=(3, 3), dilation=(1, 1),
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

        self.activation = nn.LeakyReLU(0.2, inplace=False)

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
                 kernel_size=(5, 3), num_res_blocks=2, num_groups=8, dropout=0.1):
        super().__init__()
        self.target_size = target_size

        pad_f = (kernel_size[0] - 1) // 2
        pad_t = (kernel_size[1] - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=(pad_f, pad_t))
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=False)

        self.res_blocks = nn.Sequential(*[
            ResBlock2D(out_channels, num_groups=num_groups,
                       kernel_size=(3, 3), dropout=dropout)
            for _ in range(num_res_blocks)
        ])

    def forward(self, x):
        x = F.interpolate(x, size=self.target_size, mode='nearest')
        x = self.activation(self.norm(self.conv(x)))
        if self.training:
            x = checkpoint(self.res_blocks, x, use_reentrant=False)
        else:
            x = self.res_blocks(x)
        return x


class Decoder(nn.Module):
    """Full 2D convolutional decoder for complex STFT prediction via iSTFT.

    Mirrors the encoder's 2D Conv structure. Each latent token is projected
    into a small 2D feature map (freq × 1), progressively upsampled in
    frequency and time, then stitched along time to form the full STFT.

    Architecture:
        [B, S, D] → per-token projection → [B*S, C, F_init, 1]
        → 4 Upsample2DBlocks → [B*S, C_out, n_freq, frames_per_seg]
        → stitch along time → [B, C_out, n_freq, n_stft_frames]
        → output refinement → [B, 2, n_freq, n_stft_frames]
        → iSTFT → [B, 1, target_length]
    """
    def __init__(self, d_model, num_segments, target_length,
                 channels=None, upsampling_factors=None,
                 n_fft=1024, hop_length=256, win_length=1024,
                 dropout=0.0, use_postnet=False):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        self.target_length = target_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Derived STFT dimensions
        self.n_freq_bins = n_fft // 2 + 1  # 513
        # With center=True: n_frames = padded_length // hop + 1
        # padded_length is set so n_frames = num_segments * frames_per_segment
        self.n_stft_frames = num_segments * (upsampling_factors[0] if upsampling_factors else 6)
        # Recalculate from factors
        frames_per_seg = 1
        for f in (upsampling_factors or []):
            frames_per_seg *= f
        self.frames_per_seg = frames_per_seg
        self.n_stft_frames = num_segments * frames_per_seg

        # Encoder's last spatial freq dim before pooling is ~33 (513 / 2^4 ≈ 33)
        self.f_init = 33

        # Default channels: [256, 128, 64, 32, 16]
        if channels is None:
            channels = [256, 128, 64, 32, 16]
        self.channels = channels

        # Frequency upsampling targets: 33 → 65 → 129 → 257 → 513
        freq_targets = [65, 129, 257, self.n_freq_bins]
        # Time upsampling targets per token: 1 → 1 → 2 → frames_per_seg
        # For frames_per_seg=6: [1, 1, 2, 6]
        if frames_per_seg == 6:
            time_targets = [1, 1, 2, 6]
        elif frames_per_seg == 3:
            time_targets = [1, 1, 1, 3]
        else:
            # Generic: keep time=1 until last two stages
            time_targets = [1] * (len(freq_targets) - 2) + [2, frames_per_seg]

        # Token projection: d_model → ch_init * f_init
        self.in_proj = nn.Linear(d_model, channels[0] * self.f_init)
        self.in_norm = nn.GroupNorm(8, channels[0])
        self.activation = nn.LeakyReLU(0.2, inplace=False)

        # 4 upsample stages
        self.upsample_blocks = nn.ModuleList()
        kernel_map = {
            (True, True): (5, 3),    # freq+time upsample
            (True, False): (5, 1),   # freq only
            (False, True): (1, 3),   # time only
            (False, False): (3, 3),  # no upsample
        }
        prev_freq = self.f_init
        prev_time = 1
        for i in range(len(freq_targets)):
            ft = freq_targets[i]
            tt = time_targets[i]
            freq_up = ft > prev_freq
            time_up = tt > prev_time
            k = kernel_map[(freq_up, time_up)]
            self.upsample_blocks.append(
                Upsample2DBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    target_size=(ft, tt),
                    kernel_size=k,
                    num_res_blocks=2,
                    dropout=dropout,
                )
            )
            prev_freq = ft
            prev_time = tt

        # Output refinement (operates on full stitched map)
        last_ch = channels[-1]
        self.out_refine = ResBlock2D(last_ch, kernel_size=(3, 3), dropout=dropout)
        self.out_conv = nn.Conv2d(last_ch, 2, kernel_size=(3, 3), padding=(1, 1))

        # Optional linear post-net (preserves mixing equivariance exactly)
        self.postnet = None
        if use_postnet:
            self.postnet = nn.Conv1d(1, 1, kernel_size=1025, padding=512, bias=False)

        # iSTFT window
        self.register_buffer('istft_window', torch.hann_window(win_length))

    def forward(self, z, return_stft=False):
        """
        Args:
            z: [B, S, D] latent tokens
            return_stft: if True, also return predicted STFT before iSTFT

        Returns:
            y: [B, 1, target_length] reconstructed waveform
            stft_pred: [B, 2, n_freq, n_frames] (only if return_stft=True)
        """
        B, S, D = z.shape
        assert S == self.num_segments, f"Expected {self.num_segments} segments, got {S}"

        # Token projection: [B, S, D] → [B*S, C, F_init, 1]
        x = self.in_proj(z)  # [B, S, C*F_init]
        x = x.view(B * S, self.channels[0], self.f_init, 1)
        x = self.activation(self.in_norm(x))

        # Progressive 2D upsampling (per-token)
        for block in self.upsample_blocks:
            x = block(x)
        # x: [B*S, C_last, n_freq, frames_per_seg]

        # Stitch segments along time
        x = x.view(B, S, self.channels[-1], self.n_freq_bins, self.frames_per_seg)
        x = x.permute(0, 2, 3, 1, 4)  # [B, C, F, S, T_seg]
        x = x.reshape(B, self.channels[-1], self.n_freq_bins, self.n_stft_frames)

        # Output refinement on full map
        if self.training:
            x = checkpoint(self.out_refine, x, use_reentrant=False)
        else:
            x = self.out_refine(x)

        stft_pred = self.out_conv(x)  # [B, 2, F, T]

        # iSTFT: cuFFT requires float32
        x_f = stft_pred.float()
        X = torch.complex(x_f[:, 0], x_f[:, 1])  # [B, F, T]

        # Synthesize waveform via iSTFT (center=True for COLA compliance)
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

        # Optional linear post-net
        if self.postnet is not None:
            y = y + self.postnet(y)

        assert y.shape == (B, 1, self.target_length), f"Expected (B, 1, {self.target_length}), got {y.shape}"

        if return_stft:
            return y, stft_pred
        return y

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
