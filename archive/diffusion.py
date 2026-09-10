"""DiT (Diffusion Transformer) for latent audio generation.

Generates latent tokens z ∈ R^{S×D} that decode to audio via a frozen
STFT autoencoder. Uses AdaLN-Zero conditioning on timestep and optional
stem-type class embedding (Peebles & Xie, 2023).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t, dim):
    """Sinusoidal positional embedding for diffusion timestep.

    Args:
        t: [B] integer or float timesteps
        dim: embedding dimension
    Returns:
        [B, dim] embeddings
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AdaLNModulation(nn.Module):
    """Produces 6 modulation parameters (shift, scale, gate for two sub-layers)
    from a conditioning vector via a single linear + SiLU."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init so the block starts as identity
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, c):
        """c: [B, d_model] → 6 × [B, 1, d_model]"""
        return self.proj(c).unsqueeze(1).chunk(6, dim=-1)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""

    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )
        self.adaln = AdaLNModulation(d_model)

    def forward(self, x, c):
        """
        x: [B, S, D] token sequence
        c: [B, D] conditioning vector (timestep + optional class)
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(c)

        # Self-attention with AdaLN
        h = self.norm1(x) * (1 + scale1) + shift1
        h, _ = self.attn(h, h, h)
        x = x + gate1 * h

        # MLP with AdaLN
        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.mlp(h)
        x = x + gate2 * h

        return x


# ---------------------------------------------------------------------------
# DiT Model
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """Diffusion Transformer for latent audio token generation.

    Args:
        seq_len: Number of latent tokens (S=96 for comp-24x-v6)
        d_model: Token dimension (D=96 for comp-24x-v6)
        depth: Number of DiT blocks
        n_heads: Attention heads
        mlp_ratio: MLP hidden dim = d_model * mlp_ratio
        num_classes: Number of stem classes (0 = unconditional)
        dropout: Dropout rate
    """

    def __init__(
        self,
        seq_len=96,
        d_model=96,
        depth=8,
        n_heads=6,
        mlp_ratio=4.0,
        num_classes=0,
        dropout=0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_classes = num_classes

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Optional class embedding (stem type: drums=0, bass=1, other=2, vocals=3)
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes + 1, d_model)  # +1 for unconditional (CFG)
        else:
            self.class_embed = None

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Input projection (latent tokens may need different scale than model dim)
        self.input_proj = nn.Linear(d_model, d_model)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer: LayerNorm + linear projection to output
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        nn.init.zeros_(self.final_adaln[1].weight)
        nn.init.zeros_(self.final_adaln[1].bias)

        self.output_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_t, t, y=None):
        """
        Args:
            z_t: [B, S, D] noisy latent tokens
            t: [B] diffusion timesteps (integer, 0 to T-1)
            y: [B] optional class labels (stem type indices)
        Returns:
            [B, S, D] predicted noise
        """
        # Timestep conditioning
        t_emb = sinusoidal_embedding(t, self.d_model)
        c = self.time_embed(t_emb)  # [B, D]

        # Add class conditioning if provided
        if self.class_embed is not None and y is not None:
            c = c + self.class_embed(y)

        # Input projection + positional embedding
        x = self.input_proj(z_t) + self.pos_embed

        # DiT blocks
        for block in self.blocks:
            x = block(x, c)

        # Final projection
        shift, scale = self.final_adaln(c).unsqueeze(1).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        return self.output_proj(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# DDPM Noise Scheduler
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """DDPM noise scheduler with cosine beta schedule (Nichol & Dhariwal, 2021).

    Cosine schedule provides more uniform SNR across timesteps — better for
    small datasets like MUSDB18 where every training signal matters.
    """

    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps

        # Cosine schedule
        s = 0.008  # small offset to prevent beta from being too small near t=0
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        f = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alpha_bar = alpha_bar.float()
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).float()
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).float()

        # For reverse process
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        self.posterior_variance = (betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)).float()
        self.posterior_mean_coef1 = (torch.sqrt(alpha_bar_prev) * betas / (1.0 - alpha_bar)).float()
        self.posterior_mean_coef2 = (torch.sqrt(alphas) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)).float()

    def to(self, device):
        """Move all tensors to device."""
        for attr in ['betas', 'alphas', 'alpha_bar', 'sqrt_alpha_bar',
                      'sqrt_one_minus_alpha_bar', 'posterior_variance',
                      'posterior_mean_coef1', 'posterior_mean_coef2']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(self, z_0, noise, t):
        """Forward process: q(z_t | z_0).

        Args:
            z_0: [B, S, D] clean latent tokens
            noise: [B, S, D] Gaussian noise
            t: [B] timestep indices
        Returns:
            z_t: [B, S, D] noisy latent tokens
        """
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sqrt_ab * z_0 + sqrt_1m_ab * noise

    def predict_x0(self, z_t, noise_pred, t):
        """Recover z_0 from z_t and predicted noise."""
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return (z_t - sqrt_1m_ab * noise_pred) / sqrt_ab.clamp(min=1e-8)

    @torch.no_grad()
    def ddpm_step(self, z_t, noise_pred, t):
        """One reverse DDPM step: p(z_{t-1} | z_t).

        Args:
            z_t: [B, S, D] current noisy sample
            noise_pred: [B, S, D] predicted noise from model
            t: int, current timestep (same for all in batch)
        Returns:
            z_{t-1}: [B, S, D]
        """
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        var = self.posterior_variance[t]

        # Predict x0
        x0_pred = self.predict_x0(z_t, noise_pred, torch.tensor([t], device=z_t.device))

        # Posterior mean
        mean = coef1 * x0_pred + coef2 * z_t

        if t > 0:
            noise = torch.randn_like(z_t)
            return mean + torch.sqrt(var) * noise
        else:
            return mean

    @torch.no_grad()
    def ddim_step(self, z_t, noise_pred, t, t_prev, eta=0.0):
        """One DDIM step for faster sampling.

        Args:
            z_t: [B, S, D]
            noise_pred: [B, S, D]
            t: int, current timestep
            t_prev: int, previous timestep
            eta: stochasticity (0 = deterministic DDIM, 1 = DDPM)
        """
        ab_t = self.alpha_bar[t]
        ab_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=z_t.device)

        # Predict x0
        x0_pred = self.predict_x0(z_t, noise_pred, torch.tensor([t], device=z_t.device))

        # DDIM formula
        sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
        dir_xt = torch.sqrt(1 - ab_prev - sigma ** 2) * noise_pred
        z_prev = torch.sqrt(ab_prev) * x0_pred + dir_xt

        if eta > 0 and t_prev >= 0:
            z_prev = z_prev + sigma * torch.randn_like(z_t)

        return z_prev

    @torch.no_grad()
    def sample_ddpm(self, model, shape, device, y=None, progress=False):
        """Full DDPM reverse sampling.

        Args:
            model: DiT noise prediction model
            shape: (B, S, D) shape to generate
            device: torch device
            y: [B] optional class labels
            progress: show tqdm progress bar
        Returns:
            z_0: [B, S, D] generated latent tokens
        """
        z_t = torch.randn(shape, device=device)
        timesteps = range(self.num_timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="DDPM sampling")

        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            noise_pred = model(z_t, t_batch, y)
            z_t = self.ddpm_step(z_t, noise_pred, t)

        return z_t

    @torch.no_grad()
    def sample_ddim(self, model, shape, device, num_steps=50, y=None,
                    eta=0.0, progress=False):
        """DDIM sampling with fewer steps.

        Args:
            model: DiT noise prediction model
            shape: (B, S, D)
            device: torch device
            num_steps: number of DDIM steps (e.g., 50 for fast sampling)
            y: [B] optional class labels
            eta: 0.0 = deterministic, 1.0 = stochastic (DDPM-like)
            progress: show tqdm
        Returns:
            z_0: [B, S, D]
        """
        # Uniform timestep subsequence
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(self.num_timesteps - 1, -1, -step_size))
        if progress:
            from tqdm import tqdm
            timesteps_iter = tqdm(timesteps, desc="DDIM sampling")
        else:
            timesteps_iter = timesteps

        z_t = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps_iter):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            noise_pred = model(z_t, t_batch, y)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            z_t = self.ddim_step(z_t, noise_pred, t, t_prev, eta)

        return z_t
