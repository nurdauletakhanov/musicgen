"""
Latent Diffusion Model for Music Generation

This module implements a DDPM (Denoising Diffusion Probabilistic Model) that operates
in the latent space of a pre-trained autoencoder.

Key concepts:
- Forward process: gradually add noise to data over T timesteps
- Reverse process: learn to predict and remove noise, generating new samples
- Cosine schedule: controls how much noise is added at each timestep

Usage:
    scheduler = NoiseScheduler(num_timesteps=1000)
    denoiser = Denoiser(d_model=256, num_segments=25, ...)
    
    # Training: predict noise from noisy latents
    noise = torch.randn_like(z0)
    z_noisy = scheduler.add_noise(z0, noise, t)
    noise_pred = denoiser(z_noisy, t)
    loss = F.mse_loss(noise_pred, noise)
    
    # Sampling: iteratively denoise from pure noise
    z = torch.randn(batch_size, num_segments, d_model)
    for t in reversed(range(T)):
        noise_pred = denoiser(z, t)
        z = scheduler.step(z, noise_pred, t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Noise Scheduler
# =============================================================================

class NoiseScheduler:
    """
    DDPM noise scheduler with cosine beta schedule.
    
    The scheduler controls:
    1. How much noise to add at each timestep (forward process)
    2. How to remove predicted noise (reverse process / sampling)
    
    Cosine schedule is preferred for small latent spaces as it:
    - Adds noise more gradually at the start
    - Preserves signal longer than linear schedule
    
    Args:
        num_timesteps: Total diffusion steps T (typically 1000)
        beta_start: Starting noise level (only used for linear schedule)
        beta_end: Ending noise level (only used for linear schedule)  
        schedule: 'cosine' (recommended) or 'linear'
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = 'cosine',
    ):
        self.num_timesteps = num_timesteps
        
        # Compute beta schedule (noise variance at each step)
        if schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Alpha values (signal retention)
        # alpha_t = 1 - beta_t (how much signal remains after one step)
        self.alphas = 1.0 - self.betas
        
        # Cumulative product: alpha_bar_t = product(alpha_1 ... alpha_t)
        # This lets us jump directly to any timestep: z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For sampling (reverse process)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance for sampling
        # This is the variance of p(z_{t-1} | z_t, z_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models"
        
        Creates a smoother noise schedule that:
        - Starts very slowly (preserves clean signal longer)
        - Accelerates in the middle
        - Reaches full noise by the end
        """
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps)
        
        # f(t) = cos^2((t/T + s) / (1 + s) * pi/2)
        f_t = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        
        # alpha_bar = f(t) / f(0)
        alphas_cumprod = f_t / f_t[0]
        
        # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clamp to prevent numerical issues
        return torch.clamp(betas, 0.0001, 0.999)
    
    def add_noise(
        self,
        z_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward process: add noise to clean latents.
        
        q(z_t | z_0) = N(sqrt(alpha_bar_t) * z_0, (1 - alpha_bar_t) * I)
        
        This is the "reparameterization trick":
        z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Args:
            z_0: Clean latents [B, S, D]
            noise: Gaussian noise [B, S, D]  
            timesteps: Timestep indices [B]
            
        Returns:
            z_t: Noisy latents [B, S, D]
        """
        device = z_0.device
        
        # Get coefficients for each sample in batch
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device)
        
        # Reshape for broadcasting: [B] -> [B, 1, 1]
        while sqrt_alpha.dim() < z_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        # z_t = sqrt(alpha_bar) * z_0 + sqrt(1 - alpha_bar) * noise
        z_t = sqrt_alpha * z_0 + sqrt_one_minus_alpha * noise
        
        return z_t
    
    def step(
        self,
        z_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Reverse process: one denoising step.
        
        Given z_t and predicted noise, compute z_{t-1}.
        
        The DDPM sampling formula:
        z_{t-1} = 1/sqrt(alpha_t) * (z_t - beta_t/sqrt(1-alpha_bar_t) * noise_pred) + sigma_t * noise
        
        Args:
            z_t: Current noisy latents [B, S, D]
            noise_pred: Predicted noise from denoiser [B, S, D]
            t: Current timestep (integer)
            add_noise: Whether to add stochastic noise (False for t=0)
            
        Returns:
            z_{t-1}: Less noisy latents [B, S, D]
        """
        device = z_t.device
        
        # Get precomputed values
        beta_t = self.betas[t].to(device)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].to(device)
        
        # Predict z_0 from z_t and noise_pred (not used directly, but helpful conceptually)
        # z_0_pred = (z_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
        
        # Compute mean of p(z_{t-1} | z_t)
        # mu = 1/sqrt(alpha_t) * (z_t - beta_t/sqrt(1-alpha_bar_t) * noise_pred)
        model_mean = sqrt_recip_alpha_t * (
            z_t - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t
        )
        
        if t == 0 or not add_noise:
            return model_mean
        
        # Add noise for stochasticity (important for sample diversity)
        posterior_var = self.posterior_variance[t].to(device)
        noise = torch.randn_like(z_t)
        
        return model_mean + torch.sqrt(posterior_var) * noise
    
    def to(self, device):
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


# =============================================================================
# Timestep Embedding
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal timestep embeddings as used in Transformer and DDPM.
    
    Maps scalar timestep t to a d_model-dimensional vector using sine/cosine
    functions at different frequencies. This gives the network a rich
    representation of "how noisy is this input?"
    
    The embedding is:
    - PE(t, 2i) = sin(t / 10000^(2i/d))
    - PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] integer timesteps
            
        Returns:
            embeddings: [B, d_model]
        """
        device = timesteps.device
        half_dim = self.d_model // 2
        
        # Compute frequencies: 1 / 10000^(2i/d)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
        )
        
        # timesteps: [B] -> [B, 1], freqs: [half_dim] -> [1, half_dim]
        # Result: [B, half_dim]
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        
        # Interleave sin and cos: [B, d_model]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding


class TimestepEmbedding(nn.Module):
    """
    Full timestep embedding: sinusoidal + MLP projection.
    
    The MLP allows the network to learn task-specific transformations
    of the timestep encoding.
    """
    
    def __init__(self, d_model: int, d_time: int = None):
        """
        Args:
            d_model: Model dimension (output size)
            d_time: Internal timestep dimension (defaults to 4 * d_model)
        """
        super().__init__()
        d_time = d_time or d_model * 4
        
        self.sinusoidal = SinusoidalPositionEmbedding(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_time),
            nn.SiLU(),  # SiLU (Swish) works well for diffusion
            nn.Linear(d_time, d_model),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] integer timesteps
            
        Returns:
            embeddings: [B, d_model]
        """
        x = self.sinusoidal(timesteps)
        x = self.mlp(x)
        return x


# =============================================================================
# Denoiser Network (Transformer-based)
# =============================================================================

class DenoiserBlock(nn.Module):
    """
    Single transformer block for the denoiser with timestep conditioning.
    
    Architecture:
    1. Self-attention over sequence (captures global structure)
    2. Timestep modulation via scale/shift (FiLM conditioning)
    3. Feed-forward network
    
    The timestep embedding modulates the features via:
    x = x * (1 + scale) + shift
    
    This is called "adaptive layer norm" or FiLM (Feature-wise Linear Modulation).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        # Self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Timestep conditioning: produces scale and shift for each norm
        # Output: [scale1, shift1, scale2, shift2] for the two layer norms
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 4),
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, S, D]
            t_emb: Timestep embedding [B, D]
            
        Returns:
            Output features [B, S, D]
        """
        # Get scale/shift from timestep
        t_params = self.time_mlp(t_emb)  # [B, D*4]
        scale1, shift1, scale2, shift2 = t_params.chunk(4, dim=-1)  # Each [B, D]
        
        # Reshape for broadcasting: [B, D] -> [B, 1, D]
        scale1 = scale1.unsqueeze(1)
        shift1 = shift1.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        
        # Self-attention with adaptive norm
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1  # FiLM modulation
        h, _ = self.attn(h, h, h)
        x = x + h
        
        # Feed-forward with adaptive norm
        h = self.norm2(x)
        h = h * (1 + scale2) + shift2  # FiLM modulation
        h = self.ff(h)
        x = x + h
        
        return x


class Denoiser(nn.Module):
    """
    Transformer-based denoiser for latent diffusion.
    
    Takes noisy latents z_t and timestep t, predicts the noise epsilon.
    
    Architecture:
    - Input projection
    - Positional embedding for sequence position
    - Stack of DenoiserBlocks with timestep conditioning
    - Output projection
    
    The network is designed to match the autoencoder's latent shape [B, 25, 256].
    
    Args:
        d_model: Latent dimension (should match autoencoder)
        num_segments: Number of sequence tokens (should match autoencoder)
        n_heads: Attention heads
        n_layers: Number of transformer blocks
        dropout: Dropout rate
        d_cond: Conditioning dimension (for future use, e.g., MIDI)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_segments: int = 25,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        d_cond: int = None,  # For future conditioning
    ):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(d_model)
        
        # Positional embedding for sequence
        self.pos_embed = nn.Parameter(torch.randn(1, num_segments, d_model) * 0.02)
        
        # Input projection (identity if dimensions match, but useful for future changes)
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DenoiserBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Future: conditioning projection
        self.d_cond = d_cond
        if d_cond is not None:
            self.cond_proj = nn.Linear(d_cond, d_model)
        
        # Initialize output projection to zero (start with identity mapping)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        z_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor = None,  # For future use
    ) -> torch.Tensor:
        """
        Predict noise from noisy latents.
        
        Args:
            z_noisy: Noisy latents [B, S, D]
            timesteps: Timestep indices [B] (integers 0 to T-1)
            condition: Optional conditioning [B, D_cond] (for future use)
            
        Returns:
            noise_pred: Predicted noise [B, S, D]
        """
        B, S, D = z_noisy.shape
        assert S == self.num_segments, f"Expected {self.num_segments} segments, got {S}"
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"
        
        # Embed timestep
        t_emb = self.time_embed(timesteps)  # [B, D]
        
        # Future: add conditioning to timestep embedding
        if condition is not None and self.d_cond is not None:
            cond_emb = self.cond_proj(condition)  # [B, D]
            t_emb = t_emb + cond_emb
        
        # Input projection + positional embedding
        x = self.input_proj(z_noisy)
        x = x + self.pos_embed[:, :S, :]
        
        # Transformer blocks with timestep conditioning
        for block in self.blocks:
            x = block(x, t_emb)
        
        # Output projection
        x = self.output_norm(x)
        noise_pred = self.output_proj(x)
        
        return noise_pred
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Diffusion Model Wrapper
# =============================================================================

class LatentDiffusion(nn.Module):
    """
    Complete latent diffusion model combining scheduler and denoiser.
    
    This is a convenience wrapper that handles:
    - Training: computing diffusion loss
    - Sampling: generating new latents from noise
    
    Usage:
        model = LatentDiffusion(d_model=256, num_segments=25, ...)
        
        # Training
        loss = model(z_batch)
        
        # Sampling
        z_samples = model.sample(batch_size=4)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_segments: int = 25,
        n_heads: int = 8,
        n_layers: int = 6,
        num_timesteps: int = 1000,
        dropout: float = 0.1,
        schedule: str = 'cosine',
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.d_model = d_model
        self.num_segments = num_segments
        
        self.scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            schedule=schedule,
        )
        
        self.denoiser = Denoiser(
            d_model=d_model,
            num_segments=num_segments,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
    
    def forward(self, z_0: torch.Tensor) -> dict:
        """
        Compute diffusion training loss.
        
        Args:
            z_0: Clean latents [B, S, D]
            
        Returns:
            Dictionary with 'loss' and 'noise_pred' for logging
        """
        B = z_0.size(0)
        device = z_0.device
        
        # Sample random timesteps for each sample
        timesteps = torch.randint(0, self.num_timesteps, (B,), device=device)
        
        # Sample noise
        noise = torch.randn_like(z_0)
        
        # Add noise to get z_t
        z_noisy = self.scheduler.add_noise(z_0, noise, timesteps)
        
        # Predict noise
        noise_pred = self.denoiser(z_noisy, timesteps)
        
        # Simple MSE loss (epsilon-prediction)
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise': noise,
            'timesteps': timesteps,
        }
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        device: torch.device = None,
        show_progress: bool = True,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate new latent samples via DDPM or DDIM sampling.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            show_progress: Whether to show tqdm progress bar
            use_ddim: Use DDIM sampling (more stable, recommended)
            ddim_steps: Number of DDIM steps (only used if use_ddim=True)
            eta: DDIM stochasticity (0=deterministic, 1=DDPM-like)
            
        Returns:
            z_0: Generated latents [B, S, D]
        """
        device = device or next(self.parameters()).device
        self.scheduler.to(device)
        
        # Start from pure noise
        z = torch.randn(batch_size, self.num_segments, self.d_model, device=device)
        
        if use_ddim:
            return self._sample_ddim(z, ddim_steps, eta, show_progress)
        else:
            return self._sample_ddpm(z, show_progress)
    
    @torch.no_grad()
    def _sample_ddpm(self, z: torch.Tensor, show_progress: bool) -> torch.Tensor:
        """Original DDPM sampling (can be unstable at high t)."""
        batch_size = z.size(0)
        device = z.device
        
        timesteps = list(reversed(range(self.num_timesteps)))
        if show_progress:
            try:
                from tqdm import tqdm
                timesteps = tqdm(timesteps, desc="DDPM Sampling")
            except ImportError:
                pass
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = self.denoiser(z, t_batch)
            z = self.scheduler.step(z, noise_pred, t, add_noise=(t > 0))
        
        return z
    
    @torch.no_grad()
    def _sample_ddim(
        self, 
        z: torch.Tensor, 
        num_steps: int,
        eta: float,
        show_progress: bool,
    ) -> torch.Tensor:
        """
        DDIM sampling - more stable than DDPM.
        
        Uses a subset of timesteps and a different update rule that
        directly predicts x_0 from the noise prediction.
        
        Args:
            z: Starting noise [B, S, D]
            num_steps: Number of DDIM steps (e.g., 50)
            eta: Stochasticity (0=deterministic, 1=DDPM-like)
            show_progress: Show progress bar
        """
        batch_size = z.size(0)
        device = z.device
        
        # Create DDIM timestep schedule (evenly spaced)
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]
        
        if show_progress:
            try:
                from tqdm import tqdm
                timesteps_iter = tqdm(timesteps, desc="DDIM Sampling")
            except ImportError:
                timesteps_iter = timesteps
        else:
            timesteps_iter = timesteps
        
        for i, t in enumerate(timesteps_iter):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get alpha values
            alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
            
            # Get alpha_bar for previous timestep
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.scheduler.alphas_cumprod[t_prev].to(device)
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)
            
            # Predict noise
            noise_pred = self.denoiser(z, t_batch)
            
            # Predict x_0 from noise prediction
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
            x0_pred = (z - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar.clamp(min=1e-8)
            
            # Clip x0_pred for stability
            x0_pred = torch.clamp(x0_pred, -10, 10)
            
            # DDIM update
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            sqrt_one_minus_alpha_bar_prev = torch.sqrt(1 - alpha_bar_prev)
            
            # Compute sigma for stochasticity
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred
            
            # DDIM step
            z = sqrt_alpha_bar_prev * x0_pred + dir_xt
            
            # Add noise if eta > 0
            if eta > 0 and i + 1 < len(timesteps):
                noise = torch.randn_like(z)
                z = z + sigma * noise
        
        return z
    
    def to(self, device):
        """Override to also move scheduler."""
        super().to(device)
        self.scheduler.to(device)
        return self

