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

    Supports two architectures:
      - 'wave': 1D waveform encoder + HiFi-GAN decoder (v15+)
      - 'stft': 2D STFT encoder + 2D decoder with iSTFT (v6-v14, legacy)

    The architecture is selected by model.architecture in config (default: 'stft').
    """
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    stft_cfg = cfg.get('stft', {})

    sample_rate = data_cfg.get('sample_rate', 44100)
    chunk_seconds = data_cfg.get('chunk_seconds', 1.0)
    target_length = int(sample_rate * chunk_seconds)

    architecture = model_cfg.get('architecture', 'stft')

    # STFT params (used by losses even in wave mode)
    n_fft = stft_cfg.get('n_fft', 1024)
    win_length = stft_cfg.get('win_length', n_fft)
    hop_length = stft_cfg.get('hop_length', 256)

    if architecture == 'wave':
        config = {
            'd_model': model_cfg['d_model'],
            'num_segments': model_cfg['num_segments'],
            'target_length': target_length,
            'sample_rate': sample_rate,
            'n_heads': model_cfg.get('n_heads', 4),
            'n_layers': model_cfg.get('n_layers', 0),
            'dropout': model_cfg.get('dropout', 0.0),
            # Encoder
            'encoder_strides': model_cfg['encoder_strides'],
            'encoder_channels': model_cfg['encoder_channels'],
            'encoder_dilations': model_cfg.get('encoder_dilations', [1, 3, 9]),
            'encoder_kernel_scale': model_cfg.get('encoder_kernel_scale', 2),
            # Decoder
            'decoder_channels': model_cfg['decoder_channels'],
            'decoder_resblock_kernel_sizes': model_cfg.get('decoder_resblock_kernel_sizes', [3, 7]),
            'decoder_resblock_dilations': model_cfg.get('decoder_resblock_dilations', [[1, 3], [1, 3]]),
            # Losses
            'decode_mix_weight': model_cfg.get('decode_mix_weight', 0.0),
            'latent_mix_weight': model_cfg.get('latent_mix_weight', 0.0),
            'mrstft_weight': model_cfg.get('mrstft_weight', 1.0),
            'mel_weight': model_cfg.get('mel_weight', 0.0),
            'latent_l2_weight': model_cfg.get('latent_l2_weight', 0.0),
            # Mel/loss STFT params
            'n_fft': n_fft,
            'hop_length': hop_length,
            'win_length': win_length,
        }
    else:
        # Legacy STFT architecture (v6-v14)
        n_freq_bins = n_fft // 2 + 1
        config = {
            'd_model': model_cfg['d_model'],
            'n_heads': model_cfg['n_heads'],
            'n_layers': model_cfg['n_layers'],
            'num_segments': model_cfg['num_segments'],
            'n_freq_bins': n_freq_bins,
            'target_length': target_length,
            'decode_mix_weight': model_cfg.get('decode_mix_weight', 0.0),
            'mrstft_weight': model_cfg.get('mrstft_weight', 1.0),
            'mel_weight': model_cfg.get('mel_weight', 0.0),
            'sample_rate': sample_rate,
            'dropout': model_cfg.get('dropout', 0.1),
            'n_fft': n_fft,
            'hop_length': hop_length,
            'win_length': win_length,
            'num_refine_blocks': model_cfg.get('num_refine_blocks', 1),
            'channels': model_cfg.get('channels', None),
            'encoder_channels': model_cfg.get('encoder_channels', None),
            'freq_strides': model_cfg.get('freq_strides', None),
            'time_strides': model_cfg.get('time_strides', None),
            'latent_l2_weight': model_cfg.get('latent_l2_weight', 0.0),
        }

    # MR-STFT resolution config
    if 'mrstft_ffts' in model_cfg:
        config['mrstft_ffts'] = tuple(model_cfg['mrstft_ffts'])
        config['mrstft_hops'] = tuple(model_cfg['mrstft_hops'])
        config['mrstft_wins'] = tuple(model_cfg['mrstft_wins'])

    return config


