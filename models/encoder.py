import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    """Conv2D block with GroupNorm and activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        
    def forward(self, x):
        return F.leaky_relu(self.norm(self.conv(x)), 0.2, inplace=True)


class SegmentEncoderCNN(nn.Module):
    def __init__(self, d_model, n_freq_bins):
        """
        Conv2D encoder for complex STFT segments.
        
        Treats real/imag as 2 input channels, with (frequency, time) as spatial axes.
        This respects frequency locality (harmonics are adjacent bins).
        Uses GroupNorm for training stability.
        
        Args:
            d_model: Output embedding dimension
            n_freq_bins: Number of frequency bins (n_fft // 2 + 1)
        """
        super().__init__()
        self.n_freq_bins = n_freq_bins

        # Conv2D: channels=2 (real/imag), spatial=(freq, time)
        # Kernels are (freq_size, time_size) to capture harmonic structure
        # Downsample frequency aggressively, preserve time resolution
        self.conv = nn.Sequential(
            ConvBlock2D(2, 64, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1), num_groups=8),
            ConvBlock2D(64, 128, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), num_groups=8),
            ConvBlock2D(128, 256, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), num_groups=8),
            ConvBlock2D(256, 512, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), num_groups=8),
            # Global pool over remaining (freq, time) dims
            nn.AdaptiveAvgPool2d((1, 3)),
        )
        
        self.out_norm = nn.GroupNorm(8, 512)
        self.proj = nn.Linear(512 * 3, d_model)

    def forward(self, x):
        # x: [B, S, 2, F, T_seg] - S segments, 2 channels (real/imag), F freq bins, T_seg frames
        B, S, C, F, T = x.shape

        # Merge batch and segments: [B*S, 2, F, T]
        x = x.reshape(B * S, C, F, T)
        
        # Conv2D over (freq, time) spatial dims
        h = self.conv(x)          # [B*S, 512, 1, 3]
        h = self.out_norm(h)
        h = h.flatten(1)          # [B*S, 512 * 3]
        z = self.proj(h)          # [B*S, d_model]
        z = z.view(B, S, -1)      # [B, S, d_model]
        return z
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

class GlobalEncoderTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, num_segments, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments

        self.segment_pos_embed = nn.Parameter(torch.randn(1, num_segments, d_model) * 0.02)

        # Store layers individually for gradient checkpointing
        self.layers = nn.ModuleList([
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
        self.use_checkpoint = True  # Enable gradient checkpointing to save memory

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint
        
        B, T, D = x.shape
        assert T <= self.num_segments, "Number of input segments exceeds num_segments"

        x = x + self.segment_pos_embed[:, :T, :]
        
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, num_segments, n_freq_bins, dropout=0.0):
        """
        Full encoder for complex STFT spectrograms.
        
        Uses Conv2D with:
        - channels = 2 (real/imag)
        - spatial axes = (frequency, time)
        
        This respects frequency locality (harmonics are adjacent bins).
        
        Args:
            d_model: Latent dimension
            n_heads: Transformer attention heads
            n_layers: Transformer layers
            num_segments: Number of time segments (= number of output tokens)
            n_freq_bins: Number of frequency bins (n_fft // 2 + 1)
            dropout: Dropout rate
        """
        super().__init__()
        self.num_segments = num_segments
        self.n_freq_bins = n_freq_bins
        
        self.segment_encoder = SegmentEncoderCNN(
            d_model=d_model, 
            n_freq_bins=n_freq_bins,
        )
        self.global_encoder = GlobalEncoderTransformer(
            d_model=d_model, 
            n_heads=n_heads, 
            n_layers=n_layers, 
            num_segments=num_segments,
            dropout=dropout,
        )

    def forward(self, x):
        # x: [B, 2, n_freq_bins, n_frames] - complex STFT (real, imag)
        B, C, Freq, T = x.shape
        assert C == 2, f"Expected 2 channels (real, imag), got {C}"
        assert Freq == self.n_freq_bins, f"Expected {self.n_freq_bins} frequency bins, got {Freq}"

        # Pad to make divisible by num_segments (preserves all input data)
        remainder = T % self.num_segments
        if remainder > 0:
            x = F.pad(x, (0, self.num_segments - remainder))
            T = x.shape[-1]
        frames_per_segment = T // self.num_segments

        # Reshape into (batch, num_segments, 2, n_freq_bins, frames_per_segment)
        x = x.view(B, C, Freq, self.num_segments, frames_per_segment)
        x = x.permute(0, 3, 1, 2, 4)  # [B, num_segments, 2, n_freq_bins, frames_per_segment]

        tokens = self.segment_encoder(x)
        z = self.global_encoder(tokens)
        return z

    def num_parameters(self):
        return {
            "segment_encoder": self.segment_encoder.num_parameters(),
            "global_encoder": self.global_encoder.num_parameters(),
            "total": self.segment_encoder.num_parameters() + self.global_encoder.num_parameters(),
        }
