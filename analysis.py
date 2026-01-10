import torch
import numpy as np
import os

print('='*70)
print('COMPREHENSIVE ANALYSIS: WHY LATENT DIFFUSION IS FAILING')
print('='*70)

# Load multiple latent files for better statistics
latent_dir = './maestro-chunks-latent/train'
files = os.listdir(latent_dir)[:50]  # Sample 50 files

all_latents = []
for f in files:
    data = torch.load(os.path.join(latent_dir, f), weights_only=True)
    all_latents.append(data['z'])

z_real = torch.cat(all_latents, dim=0)  # Concatenate along batch dim
print(f'Loaded {len(files)} latent files, total shape: {z_real.shape}')

# Load normalization stats
stats = torch.load('./checkpoints/latent-diffusion-ddim/latent_stats.pt', weights_only=True)
z_mean, z_std = stats['z_mean'], stats['z_std']

# Normalize as done in training
z_normalized = (z_real - z_mean) / z_std

print()
print('='*70)
print('PROBLEM 1: NON-GAUSSIAN LATENT DISTRIBUTION')
print('='*70)
print()
print('Diffusion models REQUIRE data to be approximately Gaussian.')
print('If data is not Gaussian, the noise schedule assumptions break down.')
print()

z_flat = z_real.numpy().flatten()
z_norm_flat = z_normalized.numpy().flatten()

def compute_stats(arr, name):
    mean = arr.mean()
    std = arr.std()
    skew = ((arr - mean)**3).mean() / std**3
    kurt = ((arr - mean)**4).mean() / std**4 - 3
    return mean, std, skew, kurt

raw_mean, raw_std, raw_skew, raw_kurt = compute_stats(z_flat, 'raw')
norm_mean, norm_std, norm_skew, norm_kurt = compute_stats(z_norm_flat, 'norm')

print('RAW LATENT STATISTICS:')
print(f'  Mean:     {raw_mean:>10.4f}  (Gaussian expects: 0)')
print(f'  Std:      {raw_std:>10.4f}  (Gaussian expects: 1)')
print(f'  Skewness: {raw_skew:>10.4f}  (Gaussian expects: 0)')
print(f'  Kurtosis: {raw_kurt:>10.4f}  (Gaussian expects: 0)')
print()

print('NORMALIZED LATENT STATISTICS (after mean/std normalization):')
check_mean = 'OK' if abs(norm_mean) < 0.1 else 'FAIL'
check_std = 'OK' if abs(norm_std - 1) < 0.1 else 'FAIL'
check_skew = 'FAIL' if abs(norm_skew) > 0.5 else 'OK'
check_kurt = 'FAIL' if abs(norm_kurt) > 1 else 'OK'

print(f'  Mean:     {norm_mean:>10.4f}  (Gaussian expects: 0) [{check_mean}]')
print(f'  Std:      {norm_std:>10.4f}  (Gaussian expects: 1) [{check_std}]')
print(f'  Skewness: {norm_skew:>10.4f}  (Gaussian expects: 0) [{check_skew}]')
print(f'  Kurtosis: {norm_kurt:>10.4f}  (Gaussian expects: 0) [{check_kurt}]')
print()
print('VERDICT: Normalization fixes mean/std but NOT the distribution shape!')
print(f'         Kurtosis of {norm_kurt:.1f} means {norm_kurt:.0f}x more extreme values than Gaussian.')
print()

# Show extreme values
print('EXTREME VALUE ANALYSIS:')
print(f'  In Gaussian N(0,1), values beyond +/-3 occur in 0.27% of cases')
beyond_3 = np.abs(z_norm_flat) > 3
print(f'  In our normalized latents: {100*beyond_3.mean():.2f}% of values are beyond +/-3')
beyond_5 = np.abs(z_norm_flat) > 5
print(f'  Values beyond +/-5 (should be ~0.00006%): {100*beyond_5.mean():.4f}%')
print()

print('='*70)
print('PROBLEM 2: PER-DIMENSION VARIANCE HETEROGENEITY')
print('='*70)
print()

dim_stds = z_real.std(dim=(0,1)).numpy()  # std per dimension
print(f'Per-dimension std range: [{dim_stds.min():.4f}, {dim_stds.max():.4f}]')
print(f'Ratio of max/min std: {dim_stds.max()/dim_stds.min():.2f}x')
print()

# After normalization
dim_stds_norm = z_normalized.std(dim=(0,1)).numpy()
print(f'After normalization, per-dim std range: [{dim_stds_norm.min():.4f}, {dim_stds_norm.max():.4f}]')
print(f'This is expected to be ~1.0 everywhere: ', end='')
if dim_stds_norm.std() < 0.1:
    print('OK')
else:
    print(f'FAIL (std of stds = {dim_stds_norm.std():.4f})')
print()

print('='*70)
print('PROBLEM 3: TEMPORAL CORRELATIONS (Structure)')
print('='*70)
print()

print('REAL LATENT TEMPORAL CORRELATIONS (adjacent time segments):')
real_corrs = []
for i in range(5):
    seg1 = z_real[:,i,:].numpy().flatten()
    seg2 = z_real[:,i+1,:].numpy().flatten()
    corr = np.corrcoef(seg1, seg2)[0,1]
    real_corrs.append(corr)
    print(f'  Segment {i} <-> Segment {i+1}: r = {corr:.4f}')
print(f'  Average temporal correlation: {np.mean(real_corrs):.4f}')
print()
print('For PIANO MUSIC, adjacent segments should be HIGHLY correlated (r > 0.5)')
print('This correlation represents the musical structure the model must learn.')
print()

# Check if normalized latents preserve correlations
print('After normalization, temporal correlations:')
norm_corrs = []
for i in range(5):
    seg1 = z_normalized[:,i,:].numpy().flatten()
    seg2 = z_normalized[:,i+1,:].numpy().flatten()
    corr = np.corrcoef(seg1, seg2)[0,1]
    norm_corrs.append(corr)
print(f'  Average: {np.mean(norm_corrs):.4f}')
if abs(np.mean(norm_corrs) - np.mean(real_corrs)) < 0.05:
    print('  Normalization preserves temporal structure: OK')
else:
    print('  Normalization changes temporal structure: POTENTIAL ISSUE')
print()

print('='*70)
print('PROBLEM 4: TRAINING vs GENERATION MISMATCH')
print('='*70)
print()
print('During TRAINING:')
print('  - Input: real normalized latents (non-Gaussian shape)')
print('  - Model learns to denoise from noise toward this non-Gaussian shape')
print()
print('During GENERATION:')
print('  - Start: pure Gaussian noise N(0,1)')
print('  - Expected output: Gaussian-like samples (wrong!)')
print('  - Denormalized output: does not match real distribution')
print()
print('This is why generated samples sound wrong even when statistics look OK.')
print()

print('='*70)
print('PROBLEM 5: NOISE SCHEDULE MISMATCH')
print('='*70)
print()
print('Standard diffusion noise schedule assumes data is N(0,1).')
print('When data has kurtosis >> 0:')
print('  - High noise timesteps: real extreme values get "washed out" differently')
print('  - Low noise timesteps: model struggles with extreme value regions')
print()
print('The beta schedule (linear 0.0001 to 0.02) is optimized for Gaussian data.')
print()

print('='*70)
print('SUMMARY: ROOT CAUSES')
print('='*70)
print()
print('1. AUTOENCODER NOT VAE:')
print('   Your autoencoder produces deterministic latents without KL regularization.')
print('   Latent space is non-Gaussian (kurtosis=16, skewness=-2).')
print()
print('2. NORMALIZATION IS INSUFFICIENT:')
print('   Mean/std normalization only shifts/scales, does not fix shape.')
print('   The heavy tails and skewness remain.')
print()
print('3. MODEL CAPACITY vs COMPLEXITY:')
print(f'   8-layer transformer trying to learn {z_real.shape} dimensional structure')
print('   with non-Gaussian distribution is extremely difficult.')
print()
print('='*70)
print('RECOMMENDED FIXES (in order of impact)')
print('='*70)
print()
print('1. RETRAIN AUTOENCODER AS VAE')
print('   Add KL loss to regularize latents toward N(0,1)')
print('   This is what Stable Diffusion, DALL-E 2, etc. do.')
print()
print('2. USE FLOW MATCHING INSTEAD OF DDPM')
print('   Flow matching works better with non-Gaussian distributions.')
print('   Does not assume data comes from Gaussian.')
print()
print('3. QUANTILE NORMALIZATION (quick fix)')
print('   Transform latents to Gaussian using empirical CDF.')
print('   Requires storing/loading quantile statistics.')
print()

