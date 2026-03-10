"""
STFT preprocessing for MAESTRO dataset.

Key improvements:
- Hash-based unique filenames to prevent collisions
- RMS normalization per chunk before STFT for consistent amplitude
- Float16 storage with RMS values for reconstruction

Run:
    python -m data.preprocess --config config.yaml --force
"""

import os
import argparse
import csv
import json
import hashlib
import shutil
from pathlib import Path
from typing import Tuple

import torch
import yaml
from tqdm import tqdm
import soundfile as sf


def compute_stft(
    waveform: torch.Tensor,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """
    Compute complex STFT and return real/imaginary components.
    
    Args:
        waveform: [samples] tensor (should be RMS normalized)
        window: Pre-computed Hann window tensor
        n_fft: FFT size
        hop_length: hop between frames
        
    Returns:
        stft_ri: [2, n_freq_bins, n_frames] tensor (real, imag)
    """
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window.size(0),
        window=window,
        return_complex=True,
        center=True,
    )
    return torch.stack([stft.real, stft.imag], dim=0)


def rms_normalize(waveform: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, float]:
    """
    RMS normalize waveform for consistent amplitude across tracks.
    
    Args:
        waveform: Audio tensor [samples]
        eps: Small value to prevent division by zero
        
    Returns:
        normalized: Normalized waveform
        rms: Original RMS value (for potential reconstruction)
    """
    rms = waveform.pow(2).mean().sqrt().clamp_min(eps)
    return waveform / rms, rms.item()


def make_unique_filename(audio_path: str) -> str:
    """
    Generate unique filename using path hash to prevent collisions.
    
    MAESTRO has files in different year folders that may share stems.
    Example: 2004/MIDI-Unprocessed_XP_22_R1_2004_01-02_ORIG_MID--AUDIO_22_R1_2004_01_Track01_wav.wav
             2006/MIDI-Unprocessed_XP_22_R1_2004_01-02_ORIG_MID--AUDIO_22_R1_2004_01_Track01_wav.wav
    
    Would both become the same .pt file with just stem!
    """
    rel = audio_path.replace("\\", "/")
    h = hashlib.md5(rel.encode("utf-8")).hexdigest()[:8]
    stem = Path(rel).stem
    return f"{stem}__{h}.pt"


def preprocess_maestro(
    data_dir: str,
    output_dir: str,
    chunk_seconds: float = 1.0,
    sample_rate: int = 44100,
    overlap: float = 0.0,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    force: bool = False,
):
    """
    Convert MAESTRO WAV files into RMS-normalized STFT chunks and waveforms.
    
    Each source track becomes one .pt file with:
    - x_stft: [num_chunks, 2, n_freq_bins, n_frames] in float16
    - x_wave: [num_chunks, chunk_samples] in float16 (RMS-normalized waveforms)
    - rms: [num_chunks] in float32 (RMS values for denormalization)
    
    Args:
        data_dir: Path to MAESTRO dataset
        output_dir: Where to save chunks
        chunk_seconds: Length of each chunk in seconds
        sample_rate: Target sample rate
        overlap: Overlap ratio between consecutive chunks
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        force: If True, regenerate even if files exist
    """
    chunk_samples = int(sample_rate * chunk_seconds)
    hop_samples = int(chunk_samples * (1 - overlap))
    
    n_freq_bins = n_fft // 2 + 1
    
    window = torch.hann_window(win_length)
    
    # With center=True, n_frames = L // hop + 1.
    # Pad chunks so n_frames is cleanly divisible by common segment counts.
    # For 44100 samples: 44100//256+1 = 173 (prime). Pad to 44288 → 174 = 29*6.
    natural_frames = chunk_samples // hop_length + 1
    # Round up to next multiple of lcm(2,3,6) = 6 for clean segment division
    padded_frames = ((natural_frames + 5) // 6) * 6  # 173 → 174
    padded_samples = (padded_frames - 1) * hop_length  # 173 * 256 = 44288

    # Compute actual n_frames from padded dummy chunk
    dummy_chunk = torch.zeros(padded_samples)
    dummy_stft = compute_stft(dummy_chunk, window, n_fft, hop_length)
    n_frames = dummy_stft.shape[-1]
    
    csv_path = os.path.join(data_dir, "maestro-v3.0.0.csv")
    
    # Create output directories
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"STFT Preprocessing")
    print(f"=" * 50)
    print(f"Processing {len(rows)} audio files")
    print(f"Chunk: {chunk_seconds}s ({chunk_samples} samples)")
    print(f"Overlap: {overlap * 100:.0f}%")
    print(f"STFT: n_fft={n_fft}, hop={hop_length}, win={win_length}, center=True, pad={padded_samples}")
    print(f"Output shapes: x_stft=[2, {n_freq_bins}, {n_frames}], x_wave=[{chunk_samples}]")
    print(f"Using RMS normalization + float16")
    print(f"=" * 50)
    
    chunk_counts = {"train": 0, "validation": 0, "test": 0}
    file_counts = {"train": 0, "validation": 0, "test": 0}
    
    # Load existing index if it exists to speed up startup
    index_path = os.path.join(output_dir, "index.json")
    if os.path.exists(index_path) and not force:
        with open(index_path, 'r') as f:
            index = json.load(f)
        print(f"Loaded existing index with {sum(len(v) for v in index.values())} files.")
        
        # Sync current counts with index
        for split in ["train", "validation", "test"]:
            if split in index:
                file_counts[split] = len(index[split])
                chunk_counts[split] = sum(index[split].values())
    else:
        index = {"train": {}, "validation": {}, "test": {}}
    
    for row in tqdm(rows, desc="Processing"):
        split = row["split"].lower()
        audio_path = os.path.join(data_dir, row["audio_filename"])
        
        # Use hash-based unique filename
        filename = make_unique_filename(row["audio_filename"])
        save_path = os.path.join(output_dir, split, filename)
        
        # Skip if already in index and file exists (unless force)
        if not force and split in index and filename in index[split] and os.path.exists(save_path):
            continue
            
        # Skip if file exists but not in index (load once to get count)
        if os.path.exists(save_path) and not force:
            try:
                data = torch.load(save_path, map_location="cpu", weights_only=True)
                # Handle both old format (stft) and new format (x_stft)
                if isinstance(data, dict):
                    num_chunks = data.get("x_stft", data.get("stft", data)).shape[0]
                else:
                    num_chunks = data.shape[0]
                index[split][filename] = num_chunks
                chunk_counts[split] += num_chunks
                file_counts[split] += 1
                continue
            except Exception:
                pass # If file is corrupt, re-process
        
        # Load audio
        data, sr = sf.read(audio_path, dtype='float32')
        
        # Convert to mono if stereo
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        # Resample if needed
        if sr != sample_rate:
            import torchaudio.functional as F
            waveform = torch.from_numpy(data).unsqueeze(0)
            waveform = F.resample(waveform, sr, sample_rate)
            data = waveform.squeeze(0).numpy()
        
        # Extract chunks with RMS normalization
        num_samples = len(data)
        stft_chunks = []
        wave_chunks = []
        rms_values = []
        start = 0
        
        while start + chunk_samples <= num_samples:
            chunk_np = data[start:start + chunk_samples]
            chunk_tensor = torch.from_numpy(chunk_np).float()

            # RMS normalize before STFT
            chunk_norm, rms = rms_normalize(chunk_tensor)
            # Pad to padded_samples for clean STFT frame count (padding only for STFT)
            if len(chunk_norm) < padded_samples:
                chunk_padded = torch.nn.functional.pad(chunk_norm, (0, padded_samples - len(chunk_norm)))
            else:
                chunk_padded = chunk_norm
            stft_ri = compute_stft(chunk_padded, window, n_fft, hop_length)

            stft_chunks.append(stft_ri)
            wave_chunks.append(chunk_norm)  # Save unpadded waveform
            rms_values.append(rms)
            
            start += hop_samples
        
        if not stft_chunks:
            continue
        
        # Stack and save: [num_chunks, 2, n_freq_bins, n_frames] for STFT
        # [num_chunks, chunk_samples] for waveform
        stft_stacked = torch.stack(stft_chunks, dim=0).half()
        wave_stacked = torch.stack(wave_chunks, dim=0).half()
        rms_tensor = torch.tensor(rms_values, dtype=torch.float32)
        
        # Save with RMS values for denormalization if needed
        save_data = {
            "x_stft": stft_stacked,
            "x_wave": wave_stacked,
            "rms": rms_tensor,
        }
        
        try:
            temp_path = save_path + ".tmp"
            torch.save(save_data, temp_path)
            if os.path.exists(save_path):
                os.remove(save_path)
            os.rename(temp_path, save_path)
        except (RuntimeError, OSError, IOError) as e:
            print(f"\nERROR: Failed to save {save_path}")
            print(f"Error: {type(e).__name__}: {str(e)}")
            total, used, free = shutil.disk_usage(output_dir)
            print(f"Disk: {free / (1024**3):.1f} GB free")
            raise
        
        chunk_counts[split] += len(stft_chunks)
        file_counts[split] += 1
        index[split][filename] = len(stft_chunks)
    
    # Save index
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"Train:      {chunk_counts['train']:,} chunks in {file_counts['train']} files")
    print(f"Validation: {chunk_counts['validation']:,} chunks in {file_counts['validation']} files")
    print(f"Test:       {chunk_counts['test']:,} chunks in {file_counts['test']} files")
    print(f"Total:      {sum(chunk_counts.values()):,} chunks")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MAESTRO into STFT chunks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config YAML path")
    parser.add_argument("--force", action="store_true", help="Regenerate all files")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['data']
    stft_cfg = config['stft']
    
    preprocess_maestro(
        data_dir=data_cfg['data_dir'],
        output_dir=data_cfg['chunks_dir'],
        chunk_seconds=data_cfg['chunk_seconds'],
        sample_rate=data_cfg['sample_rate'],
        overlap=data_cfg.get('overlap', 0.0),
        n_fft=stft_cfg['n_fft'],
        hop_length=stft_cfg['hop_length'],
        win_length=stft_cfg['win_length'],
        force=args.force,
    )

