import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual block with Conv2D and GroupNorm."""
    def __init__(self, channels, num_groups=8, kernel_size=5, dilation=1, res_scale=0.1, dropout=0.1):
        super().__init__()
        self.res_scale = res_scale
        pad = dilation * ((kernel_size - 1) // 2)
        self.dropout = nn.Dropout1d(dropout)
        
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation)
        

        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation)

        self.activation = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return x + self.res_scale * h


class UpsampleBlock(nn.Module):
    """Upsample block with interpolate"""
    def __init__(self, in_channels, out_channels, num_groups=8, dropout=0.1, factor=8,use_res=True):
        super().__init__()

        assert out_channels % num_groups == 0, f"Channels must be divisible by num_groups, got {out_channels} % {num_groups} = {out_channels % num_groups}"


        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        self.res = nn.Sequential(
            ResBlock(out_channels, num_groups=num_groups, kernel_size=3, dilation=1, dropout=dropout),
            ResBlock(out_channels, num_groups=num_groups, kernel_size=3, dilation=3, dropout=dropout),
            ResBlock(out_channels, num_groups=num_groups, kernel_size=3, dilation=9, dropout=dropout),
        )
        self.factor = factor
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='nearest')
        x = self.activation(self.norm(self.conv(x)))
        x = self.res(x)
        return x


class DecoderWaveform(nn.Module):
    def __init__(self, d_model, num_segments, channels, upsampling_factors, target_length, dropout=0.0):
        f"""
        z: [B, num_segments, d_model] -> waveform: [B, 1, target_length]
        """
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        self.channels = channels
        self.target_length = target_length
        # Project each token to a 1D feature map: [B, S, d_model] -> [B, C, S]
        self.in_proj = nn.Linear(d_model, self.channels[0])
        self.in_norm = nn.GroupNorm(8, self.channels[0])
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        
        # Frequency upsampling blocks: each doubles frequency dimension only (factor=8)
        self.time_upsampling_blocks = nn.ModuleList([
            UpsampleBlock(self.channels[i], self.channels[i+1], dropout=dropout, use_res=True, factor=upsampling_factors[i]) for i in range(len(self.channels) - 1)
        ])

        self.out_refine = ResBlock(self.channels[-1], dropout=dropout)
        self.out_conv = nn.Conv1d(self.channels[-1], 1, kernel_size=7, padding=3)
        self.out_activation = nn.Identity()

    def forward(self, z):
        # z: [B, num_segments, d_model]
        B, S, D = z.shape
    
        assert S == self.num_segments, f"Expected {self.num_segments} segments, got {S}"

        # Project to 1D feature map
        x = self.in_proj(z)  # [B, S, channels[0]]
        x = x.transpose(1, 2)  # [B, channels[0], S]
        x = self.activation(self.in_norm(x))

        for block in self.time_upsampling_blocks:
            x = block(x)
        
        x = self.out_refine(x)
        x = self.out_conv(x)
        y = self.out_activation(x)
        
        assert y.shape == (B, 1, self.target_length), f"Expected (B, 1, {self.target_length}), got {y.shape}"

        return y

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

# Alias for backward compatibility
Decoder = DecoderWaveform