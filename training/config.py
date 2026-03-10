"""Configuration loading and management utilities."""

import os
import shutil

import torch
import yaml


def get_device(gpu_index: int = 0) -> torch.device:
    """Get CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        if gpu_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested GPU {gpu_index} but only {torch.cuda.device_count()} available."
            )
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Recursively merge override_config into base_config."""
    result = base_config.copy()
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, base_path: str = None) -> dict:
    """
    Load config file, optionally merging with a base config.
    
    Args:
        config_path: Path to the experiment config file
        base_path: Optional path to base config (defaults to configs/base.yaml)
    
    Returns:
        Merged configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Try to load and merge base config
    if base_path is None:
        # Look for base.yaml relative to the config file
        config_dir = os.path.dirname(config_path)
        if os.path.basename(config_dir) == "experiments":
            base_path = os.path.join(os.path.dirname(config_dir), "base.yaml")
        else:
            # Fallback: look in configs/base.yaml from project root
            base_path = "configs/base.yaml"
    
    if os.path.exists(base_path):
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        config = merge_configs(base_config, config)
    
    return config


def copy_config_to_checkpoint(config_path: str, save_path: str):
    """Copy the config file to the checkpoint directory for reproducibility."""
    dest_path = os.path.join(save_path, "config.yaml")
    if not os.path.exists(dest_path):
        shutil.copy2(config_path, dest_path)


def build_model_config(cfg: dict) -> dict:
    """
    Build model configuration dictionary from full config.
    
    Computes derived parameters like n_freq_bins and target_length from
    STFT and data config sections.
    
    Args:
        cfg: Full configuration dictionary with 'model', 'data', and 'stft' sections
    
    Returns:
        Model configuration dictionary ready for Autoencoder initialization
    
    Raises:
        ValueError: If upsampling_factors is missing or has wrong length
    """
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    stft_cfg = cfg.get('stft', {})
    
    # Compute STFT-derived params
    n_fft = stft_cfg.get('n_fft', 1024)
    win_length = stft_cfg.get('win_length', n_fft)
    hop_length = stft_cfg.get('hop_length', 256)
    sample_rate = data_cfg.get('sample_rate', 44100)
    chunk_seconds = data_cfg.get('chunk_seconds', 1.0)
    
    n_freq_bins = n_fft // 2 + 1
    chunk_samples = int(sample_rate * chunk_seconds)
    target_length = chunk_samples
    
    num_segments = model_cfg['num_segments']
    channels = model_cfg.get('channels', [])
    upsampling_factors = model_cfg.get('upsampling_factors', [])
    if len(channels) > 1 and len(upsampling_factors) != len(channels) - 1:
        raise ValueError(
            f"upsampling_factors length ({len(upsampling_factors)}) "
            f"must match len(channels)-1 ({len(channels) - 1})"
        )

    return {
        'd_model': model_cfg['d_model'],
        'n_heads': model_cfg['n_heads'],
        'n_layers': model_cfg['n_layers'],
        'num_segments': num_segments,
        'n_freq_bins': n_freq_bins,
        'channels': channels,
        'upsampling_factors': upsampling_factors,
        'target_length': target_length,
        'latent_mix_weight': model_cfg.get('latent_mix_weight', 0.0),
        'decode_mix_weight': model_cfg.get('decode_mix_weight', 0.0),
        'mrstft_weight': model_cfg.get('mrstft_weight', 1.0),
        'l1_weight': model_cfg.get('l1_weight', 1.0),
        'stft_loss_weight': model_cfg.get('stft_loss_weight', 0.0),
        'mix_l1_weight': model_cfg.get('mix_l1_weight', 1.0),
        'mix_mrstft_weight': model_cfg.get('mix_mrstft_weight', 1.0),
        'dropout': model_cfg.get('dropout', 0.1),
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length,
    }


