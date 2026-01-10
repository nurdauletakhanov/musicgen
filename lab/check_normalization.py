"""
Check normalization statistics on MAESTRO chunk STFT data.

This script loads chunks from the dataset and prints statistics to verify:
1. STFT data distribution (x_stft)
2. Waveform RMS normalization (x_wave should have RMS ≈ 1.0)
"""

import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import data modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import STFTChunkDataset


def compute_statistics(tensor: torch.Tensor, name: str):
    """Compute and return statistics for a tensor."""
    tensor_np = tensor.detach().cpu().float().numpy()
    
    stats = {
        "name": name,
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "mean": float(tensor_np.mean()),
        "std": float(tensor_np.std()),
        "min": float(tensor_np.min()),
        "max": float(tensor_np.max()),
        "median": float(np.median(tensor_np)),
    }
    
    # For 2D+ tensors, also compute per-channel statistics
    if tensor.ndim >= 2:
        # Flatten all but first dimension
        flat = tensor_np.reshape(tensor_np.shape[0], -1)
        stats["mean_per_sample"] = {
            "mean": float(flat.mean(axis=1).mean()),
            "std": float(flat.mean(axis=1).std()),
            "min": float(flat.mean(axis=1).min()),
            "max": float(flat.mean(axis=1).max()),
        }
    
    return stats


def compute_rms_statistics(waveform: torch.Tensor):
    """Compute RMS statistics for waveform chunks."""
    # waveform shape: [chunk_samples]
    # Compute RMS for this chunk
    rms = waveform.pow(2).mean().sqrt()
    
    # Also compute per-sample statistics
    waveform_np = waveform.detach().cpu().float().numpy()
    
    return {
        "rms": float(rms.item()),
        "mean": float(waveform_np.mean()),
        "std": float(waveform_np.std()),
        "min": float(waveform_np.min()),
        "max": float(waveform_np.max()),
    }


def print_statistics(stats_list, title="Statistics"):
    """Print statistics in a readable format."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    for stats in stats_list:
        print(f"\n{stats['name']}")
        print(f"  Shape: {stats['shape']}")
        print(f"  Dtype: {stats['dtype']}")
        print(f"  Mean:  {stats['mean']:>12.6f}")
        print(f"  Std:   {stats['std']:>12.6f}")
        print(f"  Min:   {stats['min']:>12.6f}")
        print(f"  Max:   {stats['max']:>12.6f}")
        print(f"  Median:{stats['median']:>12.6f}")
        
        if "mean_per_sample" in stats:
            mp = stats["mean_per_sample"]
            print(f"  Per-sample mean stats:")
            print(f"    Mean: {mp['mean']:>12.6f} ± {mp['std']:>12.6f}")
            print(f"    Range: [{mp['min']:>12.6f}, {mp['max']:>12.6f}]")


def main():
    # Load config
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found!")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    chunks_dir = config['data']['chunks_dir']
    index_path = os.path.join(chunks_dir, "index.json")
    
    if not os.path.exists(index_path):
        print(f"Error: index.json not found at {index_path}")
        print("Please run preprocessing first: python -m data.preprocess --config config.yaml")
        sys.exit(1)
    
    # Load dataset for train split
    print("Loading dataset...")
    dataset = STFTChunkDataset(
        chunks_dir=chunks_dir,
        index_path=index_path,
        split="train",
        dtype=torch.float32,
    )
    
    print(f"Dataset loaded: {len(dataset):,} chunks")
    
    # Sample chunks for statistics (sample a reasonable number)
    num_samples = min(1000, len(dataset))  # Sample up to 1000 chunks
    sample_indices = torch.randperm(len(dataset))[:num_samples].tolist()
    
    print(f"\nSampling {num_samples} chunks for statistics...")
    
    # Accumulate statistics
    all_stft_real = []
    all_stft_imag = []
    all_waveforms = []
    all_rms_values = []
    
    for idx in tqdm(sample_indices, desc="Processing chunks"):
        sample = dataset[idx]
        x_stft = sample["x_stft"]  # [2, n_freq_bins, n_frames]
        x_wave = sample["x_wave"]  # [chunk_samples]
        
        # Separate real and imaginary parts
        stft_real = x_stft[0]  # [n_freq_bins, n_frames]
        stft_imag = x_stft[1]  # [n_freq_bins, n_frames]
        
        all_stft_real.append(stft_real)
        all_stft_imag.append(stft_imag)
        all_waveforms.append(x_wave)
        
        # Compute RMS for this chunk
        rms = x_wave.pow(2).mean().sqrt()
        all_rms_values.append(rms.item())
    
    # Concatenate all samples
    print("\nComputing aggregate statistics...")
    
    stft_real_cat = torch.stack(all_stft_real, dim=0)  # [num_samples, n_freq_bins, n_frames]
    stft_imag_cat = torch.stack(all_stft_imag, dim=0)  # [num_samples, n_freq_bins, n_frames]
    waveforms_cat = torch.stack(all_waveforms, dim=0)   # [num_samples, chunk_samples]
    
    # Compute statistics
    stats_list = [
        compute_statistics(stft_real_cat, "STFT Real Part"),
        compute_statistics(stft_imag_cat, "STFT Imaginary Part"),
        compute_statistics(waveforms_cat, "Waveforms (RMS-normalized)"),
    ]
    
    # Print statistics
    print_statistics(stats_list, "Data Normalization Statistics")
    
    # Print RMS statistics
    print(f"\n{'='*70}")
    print("RMS Normalization Check")
    print(f"{'='*70}")
    rms_array = np.array(all_rms_values)
    print(f"Number of chunks analyzed: {len(all_rms_values)}")
    print(f"RMS Mean:   {rms_array.mean():>12.6f}")
    print(f"RMS Std:    {rms_array.std():>12.6f}")
    print(f"RMS Min:    {rms_array.min():>12.6f}")
    print(f"RMS Max:    {rms_array.max():>12.6f}")
    print(f"RMS Median: {np.median(rms_array):>12.6f}")
    
    # Check if RMS is close to 1.0 (which would indicate proper normalization)
    rms_mean = rms_array.mean()
    rms_std = rms_array.std()
    print(f"\nExpected RMS ≈ 1.0 for normalized waveforms")
    print(f"Actual RMS Mean: {rms_mean:.6f} (difference from 1.0: {abs(rms_mean - 1.0):.6f})")
    
    if abs(rms_mean - 1.0) < 0.1 and rms_std < 0.5:
        print("✓ RMS normalization appears correct (mean close to 1.0, low variance)")
    else:
        print("⚠ Warning: RMS normalization may not be correct")
    
    # Additional check: compute magnitude from STFT
    print(f"\n{'='*70}")
    print("STFT Magnitude Statistics")
    print(f"{'='*70}")
    stft_magnitude = torch.sqrt(stft_real_cat.pow(2) + stft_imag_cat.pow(2))
    mag_stats = compute_statistics(stft_magnitude, "STFT Magnitude")
    print_statistics([mag_stats], "STFT Magnitude")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

