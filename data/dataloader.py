import os
import json
from bisect import bisect_right
from typing import Optional, Tuple, Iterator, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class STFTChunkDataset(Dataset):
    """Chunk-level dataset for preprocessed MAESTRO STFT tensors and waveforms."""

    def __init__(
        self,
        chunks_dir: str,
        index_path: str,
        split: str,
        dtype: torch.dtype = torch.float32,  # Use float32 by default (convert on CPU, not GPU)
        cache_last_file: bool = True,  # Now default True since we access sequentially
    ):
        """
        Args:
            chunks_dir: Base directory containing split subfolders with .pt chunk files
            index_path: Path to index.json produced by preprocessing
            split: Which split to load (e.g., 'train', 'validation')
            dtype: Desired dtype for returned chunks
            cache_last_file: Cache the last loaded file (recommended with ShardedSampler)
        
        Returns:
            Dict with keys 'x_stft' (STFT tensor) and 'x_wave' (waveform tensor)
        """
        self.chunks_dir = chunks_dir
        self.split = split
        self.dtype = dtype
        self.cache_last_file = cache_last_file

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"index.json not found at {index_path}. Run data preprocessing first.")

        with open(index_path, "r") as f:
            index = json.load(f)

        if split not in index:
            raise ValueError(f"Split '{split}' not found in index.json. Available: {list(index.keys())}")

        split_index = index[split]
        if not isinstance(split_index, dict) or len(split_index) == 0:
            raise RuntimeError(f"No entries found for split '{split}' in index.json.")

        self.files = []
        total = 0
        missing = []

        # Sort for deterministic ordering
        for filename, count in sorted(split_index.items()):
            path = os.path.join(chunks_dir, split, filename)
            if not os.path.exists(path):
                missing.append(path)
                continue

            count = int(count)
            start = total
            total += count
            self.files.append({"path": path, "count": count, "start": start, "end": total})

        if missing:
            raise FileNotFoundError(
                f"Missing chunk files for split '{split}': {missing[:3]}{'...' if len(missing) > 3 else ''}"
            )

        if total == 0:
            raise RuntimeError(f"Split '{split}' contains zero chunks.")

        self.total_chunks = total
        self._ends = [f["end"] for f in self.files]
        self.file_count = len(self.files)

        # Cache for sequential access
        self._cache_path = None
        self._cache_data = None

    def __len__(self):
        return self.total_chunks

    def _load_file(self, path: str) -> Dict[str, torch.Tensor]:
        if self.cache_last_file and path == self._cache_path and self._cache_data is not None:
            return self._cache_data

        data = torch.load(path, map_location="cpu", weights_only=True)
        
        # Handle both old format (stft) and new format (x_stft, x_wave)
        if isinstance(data, dict):
            # New format with x_stft and x_wave
            if "x_stft" in data and "x_wave" in data:
                result = {"x_stft": data["x_stft"], "x_wave": data["x_wave"]}
            # Old format with just stft - need to reprocess
            elif "stft" in data:
                raise RuntimeError(
                    f"Old format detected in {path}. File contains 'stft' but not 'x_stft' and 'x_wave'. "
                    f"Please reprocess the data by running: python -m data.preprocess --config config.yaml --force"
                )
            else:
                raise ValueError(
                    f"Unexpected data format in {path}. Expected dict with 'x_stft' and 'x_wave' keys."
                )
        else:
            raise RuntimeError(
                f"Old format detected in {path}. File is a tensor, not a dict with 'x_stft' and 'x_wave'. "
                f"Please reprocess the data by running: python -m data.preprocess --config config.yaml --force"
            )

        if self.cache_last_file:
            self._cache_path = path
            self._cache_data = result
        return result

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.total_chunks:
            raise IndexError(f"Index {idx} out of range for dataset length {self.total_chunks}")

        file_idx = bisect_right(self._ends, idx)
        start = 0 if file_idx == 0 else self._ends[file_idx - 1]
        local_idx = idx - start

        file_entry = self.files[file_idx]
        file_data = self._load_file(file_entry["path"])
        
        x_stft = file_data["x_stft"][local_idx].to(self.dtype).contiguous()
        x_wave = file_data["x_wave"][local_idx].to(self.dtype).contiguous()
        
        return {"x_stft": x_stft, "x_wave": x_wave}


class ShardedSampler(Sampler[int]):
    """
    Sampler that shuffles at file (shard) level, then iterates chunks within each file.
    
    This prevents disk thrashing by ensuring sequential access within files.
    Each file is loaded once per epoch, and chunks within are accessed in order
    (or shuffled within file if shuffle_within_file=True).
    """

    def __init__(
        self,
        dataset: STFTChunkDataset,
        shuffle_files: bool = True,
        shuffle_within_file: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.shuffle_files = shuffle_files
        self.shuffle_within_file = shuffle_within_file
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across epochs."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.dataset.total_chunks

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Get file order
        file_indices = list(range(self.dataset.file_count))
        if self.shuffle_files:
            file_indices = torch.randperm(len(file_indices), generator=g).tolist()

        indices: List[int] = []
        for file_idx in file_indices:
            file_entry = self.dataset.files[file_idx]
            start = file_entry["start"]
            count = file_entry["count"]
            
            # Chunk indices within this file
            chunk_indices = list(range(start, start + count))
            
            if self.shuffle_within_file:
                # Shuffle within file
                perm = torch.randperm(len(chunk_indices), generator=g).tolist()
                chunk_indices = [chunk_indices[i] for i in perm]
            
            indices.extend(chunk_indices)

        return iter(indices)


def build_dataloaders(
    chunks_dir: str,
    train_split: str,
    val_split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    latent_mix_weight: float,
    decode_mix_weight: float = 0.0,
    logger: Optional[object] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, STFTChunkDataset, STFTChunkDataset, ShardedSampler]:
    """
    Create train/val datasets and dataloaders with sharded sampling.
    
    Returns:
        train_loader, val_loader, train_dataset, val_dataset, train_sampler
        (train_sampler returned so you can call set_epoch() each epoch)
    """
    index_path = os.path.join(chunks_dir, "index.json")
    # Both mixing losses require even batch size
    drop_last = latent_mix_weight > 0.0 or decode_mix_weight > 0.0
    
    if pin_memory is None:
        pin_memory = device.type == "cuda"
    if persistent_workers is None:
        persistent_workers = pin_memory and num_workers > 0

    # Cache is beneficial with sharded sampler (sequential file access)
    train_dataset = STFTChunkDataset(
        chunks_dir=chunks_dir,
        index_path=index_path,
        split=train_split,
        cache_last_file=True,
    )
    val_dataset = STFTChunkDataset(
        chunks_dir=chunks_dir,
        index_path=index_path,
        split=val_split,
        cache_last_file=True,
    )

    # Sharded sampler for train (shuffles files, then chunks within files)
    train_sampler = ShardedSampler(
        train_dataset,
        shuffle_files=True,
        shuffle_within_file=True,
    )

    # Build loader kwargs
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,  # Use sharded sampler instead of shuffle=True
        **loader_kwargs,
    )

    # Validation: no shuffle, sequential access
    # Also need to drop last batch if mixing losses require even batch size
    val_loader_kwargs = loader_kwargs.copy()
    val_loader_kwargs["drop_last"] = drop_last  # Use same drop_last as train when mixing is enabled
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **val_loader_kwargs,
    )

    if logger:
        logger.info(
            f"Train split '{train_split}': {len(train_dataset):,} chunks across {train_dataset.file_count} files"
        )
        logger.info(
            f"Val split '{val_split}': {len(val_dataset):,} chunks across {val_dataset.file_count} files"
        )
        logger.info(f"Using ShardedSampler for sequential file access (reduces disk thrashing)")

    return train_loader, val_loader, train_dataset, val_dataset, train_sampler
