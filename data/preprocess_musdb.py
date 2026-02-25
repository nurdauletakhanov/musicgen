"""
STFT preprocessing for MUSDB18 dataset.

Extracts individual stems from .stem.mp4 files and produces .pt shard files
with the same format as the MAESTRO preprocessor, plus stem-aware indexing
to enable paired stem loading for mixing equivariance training.

Stems per track:
  0: mixture (skipped — redundant, equals sum of 1-4)
  1: drums
  2: bass
  3: other (accompaniment)
  4: vocals

Requirements:
    pip install stempeg
    conda install -c conda-forge ffmpeg

Run:
    python -m data.preprocess_musdb --config configs/base.yaml --force
"""

import os
import argparse
import json
import hashlib
import shutil
from pathlib import Path
from typing import List

import torch
import numpy as np
import yaml
from tqdm import tqdm

from data.preprocess import compute_stft, rms_normalize


STEM_NAMES = {
    0: "mixture",
    1: "drums",
    2: "bass",
    3: "other",
    4: "vocals",
}


def make_track_hash(track_path: str) -> str:
    """Generate a short hash for a track path to ensure unique filenames."""
    rel = track_path.replace("\\", "/")
    return hashlib.md5(rel.encode("utf-8")).hexdigest()[:8]


def load_stems(track_path: str, sample_rate: int = 44100) -> np.ndarray:
    """
    Load all stems from a .stem.mp4 file using stempeg.

    Returns:
        stems: np.ndarray [num_stems, num_samples, num_channels]
               Typically [5, N, 2] for stereo stems.
    """
    import stempeg

    stems, sr = stempeg.read_stems(track_path, sample_rate=sample_rate)
    return stems


def preprocess_musdb(
    musdb_dir: str,
    output_dir: str,
    stems: List[int] = None,
    chunk_seconds: float = 1.0,
    sample_rate: int = 44100,
    overlap: float = 0.0,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    min_rms: float = 1e-4,
    force: bool = False,
):
    """
    Convert MUSDB18 .stem.mp4 files into RMS-normalized STFT chunks.

    Each stem of each track becomes one .pt file with:
    - x_stft: [num_chunks, 2, n_freq_bins, n_frames] in float16
    - x_wave: [num_chunks, chunk_samples] in float16 (RMS-normalized)
    - rms: [num_chunks] in float32

    The index.json groups stems by track for aligned pair loading.

    Args:
        musdb_dir: Path to musdb18/ directory (containing train/ and test/)
        output_dir: Where to save preprocessed chunks
        stems: Which stem indices to extract [1,2,3,4] by default (skip mixture)
        chunk_seconds: Length of each chunk in seconds
        sample_rate: Target sample rate
        overlap: Overlap ratio between consecutive chunks
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        min_rms: Minimum RMS threshold — chunks below this are near-silent and skipped
        force: If True, regenerate even if files exist
    """
    if stems is None:
        stems = [1, 2, 3, 4]

    chunk_samples = int(sample_rate * chunk_seconds)
    hop_samples = int(chunk_samples * (1 - overlap))

    n_freq_bins = n_fft // 2 + 1
    window = torch.hann_window(win_length)

    # Compute actual n_frames
    dummy_stft = compute_stft(torch.zeros(chunk_samples), window, n_fft, hop_length)
    n_frames = dummy_stft.shape[-1]

    # Create output directories
    for split in ["train", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Load existing index if resuming
    index_path = os.path.join(output_dir, "index.json")
    if os.path.exists(index_path) and not force:
        with open(index_path, "r") as f:
            index = json.load(f)
        print(f"Loaded existing index.")
    else:
        index = {"train": {}, "test": {}}

    print(f"MUSDB18 STFT Preprocessing")
    print(f"=" * 50)
    print(f"Stems: {[STEM_NAMES[s] for s in stems]}")
    print(f"Chunk: {chunk_seconds}s ({chunk_samples} samples)")
    print(f"Overlap: {overlap * 100:.0f}%")
    print(f"STFT: n_fft={n_fft}, hop={hop_length}, win={win_length}, center=False")
    print(f"Output shapes: x_stft=[2, {n_freq_bins}, {n_frames}], x_wave=[{chunk_samples}]")
    print(f"Min RMS threshold: {min_rms}")
    print(f"=" * 50)

    stats = {"train": {"tracks": 0, "chunks": 0}, "test": {"tracks": 0, "chunks": 0}}

    for split in ["train", "test"]:
        split_dir = os.path.join(musdb_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: {split_dir} not found, skipping.")
            continue

        # Find all .stem.mp4 files
        track_files = sorted([
            f for f in os.listdir(split_dir) if f.endswith(".stem.mp4")
        ])

        print(f"\n{split}: {len(track_files)} tracks")

        for track_file in tqdm(track_files, desc=f"Processing {split}"):
            track_path = os.path.join(split_dir, track_file)
            track_name = track_file.replace(".stem.mp4", "")
            track_hash = make_track_hash(os.path.join(split, track_file))
            track_key = f"{track_name}__{track_hash}"

            # Check if already processed
            if not force and track_key in index[split]:
                existing = index[split][track_key]
                # Verify all stem files exist
                all_exist = all(
                    os.path.exists(os.path.join(output_dir, split, stem_file))
                    for stem_file in existing.get("stems", {}).values()
                )
                if all_exist:
                    stats[split]["tracks"] += 1
                    stats[split]["chunks"] += existing.get("num_chunks", 0)
                    continue

            # Load all stems at once (more efficient than per-stem loading)
            try:
                all_stems = load_stems(track_path, sample_rate)
            except Exception as e:
                print(f"\nError loading {track_file}: {e}")
                continue

            # all_stems shape: [num_stems, num_samples, num_channels]
            # Convert stereo to mono: average channels
            # Result: [num_stems, num_samples]
            if all_stems.ndim == 3:
                all_stems_mono = all_stems.mean(axis=2)
            else:
                all_stems_mono = all_stems

            num_samples = all_stems_mono.shape[1]

            # Determine valid chunk positions (shared across all stems)
            chunk_starts = []
            pos = 0
            while pos + chunk_samples <= num_samples:
                chunk_starts.append(pos)
                pos += hop_samples

            if not chunk_starts:
                print(f"\nSkipping {track_name}: too short ({num_samples} samples)")
                continue

            # Find chunks where ALL requested stems have sufficient energy
            # This ensures alignment: same chunk indices across all stems
            valid_chunks = []
            for ci, start in enumerate(chunk_starts):
                all_above_threshold = True
                for stem_idx in stems:
                    chunk_np = all_stems_mono[stem_idx, start:start + chunk_samples]
                    rms_val = np.sqrt(np.mean(chunk_np ** 2))
                    if rms_val < min_rms:
                        all_above_threshold = False
                        break
                if all_above_threshold:
                    valid_chunks.append(ci)

            if not valid_chunks:
                print(f"\nSkipping {track_name}: no valid chunks above RMS threshold")
                continue

            # Process each stem
            stem_files = {}
            for stem_idx in stems:
                stem_name = STEM_NAMES[stem_idx]
                filename = f"{track_name}__{track_hash}__stem{stem_idx}.pt"
                save_path = os.path.join(output_dir, split, filename)

                stft_chunks = []
                wave_chunks = []
                rms_values = []

                for ci in valid_chunks:
                    start = chunk_starts[ci]
                    chunk_np = all_stems_mono[stem_idx, start:start + chunk_samples]
                    chunk_tensor = torch.from_numpy(chunk_np.copy()).float()

                    # RMS normalize
                    chunk_norm, rms = rms_normalize(chunk_tensor)
                    stft_ri = compute_stft(chunk_norm, window, n_fft, hop_length)

                    stft_chunks.append(stft_ri)
                    wave_chunks.append(chunk_norm)
                    rms_values.append(rms)

                # Stack and save
                stft_stacked = torch.stack(stft_chunks, dim=0).half()
                wave_stacked = torch.stack(wave_chunks, dim=0).half()
                rms_tensor = torch.tensor(rms_values, dtype=torch.float32)

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

                stem_files[stem_name] = filename

            # Update index
            num_chunks = len(valid_chunks)
            index[split][track_key] = {
                "num_chunks": num_chunks,
                "stems": stem_files,
            }

            stats[split]["tracks"] += 1
            stats[split]["chunks"] += num_chunks

        # Save index after each split (for crash recovery)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    # Final summary
    print(f"\nPreprocessing complete!")
    print(f"Train: {stats['train']['chunks']:,} chunks from {stats['train']['tracks']} tracks")
    print(f"Test:  {stats['test']['chunks']:,} chunks from {stats['test']['tracks']} tracks")
    total_chunks = stats['train']['chunks'] + stats['test']['chunks']
    print(f"Total: {total_chunks:,} chunks")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MUSDB18 into STFT chunks")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config YAML path")
    parser.add_argument("--musdb-dir", type=str, default="./musdb18", help="Path to musdb18 directory")
    parser.add_argument("--output-dir", type=str, default="./musdb-chunks-stft", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Regenerate all files")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    stft_cfg = config["stft"]
    data_cfg = config.get("data", {})

    preprocess_musdb(
        musdb_dir=args.musdb_dir,
        output_dir=args.output_dir,
        chunk_seconds=data_cfg.get("chunk_seconds", 1.0),
        sample_rate=data_cfg.get("sample_rate", 44100),
        overlap=data_cfg.get("overlap", 0.0),
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg["win_length"],
        force=args.force,
    )
