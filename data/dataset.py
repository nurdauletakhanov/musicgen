"""
Waveform dataset for v1.

Schema (see data/preprocess.py for how files are produced):

  <chunks_dir>/
    index.json              {"train": {key: entry}, "test": {key: entry}}
    train/<key>.pt          {"x_wave": fp16 [N, L], "peak": fp32 [N]}
    test/<key>.pt

  entry: {"filename": str, "num_chunks": int, "source": "musdb"|"maestro"|"fma"}

The dataset is single-source-per-track — every .pt file holds exactly one stem
(the mixture). No stem-pair logic, no STFT tensor in storage.
"""

import json
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class WaveformDataset(Dataset):
    """
    Random-access dataset over (track, chunk) pairs in one split of a
    unified chunks directory.

    __getitem__(idx) -> {"x_wave": [1, L] fp32, "source": str}
    """

    def __init__(
        self,
        chunks_dir: str,
        split: str,
        cache_size: int = 8,
    ):
        index_path = os.path.join(chunks_dir, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing {index_path}; run preprocessing first.")

        with open(index_path, "r") as f:
            index = json.load(f)

        if split not in index:
            raise ValueError(f"split '{split}' not in index; have {list(index.keys())}")

        self.chunks_dir = chunks_dir
        self.split = split
        self.files: List[Dict] = []
        total = 0
        for key, entry in sorted(index[split].items()):
            num = int(entry["num_chunks"])
            if num <= 0:
                continue
            path = os.path.join(chunks_dir, split, entry["filename"])
            if not os.path.exists(path):
                continue
            self.files.append({
                "key": key,
                "path": path,
                "source": entry.get("source", "unknown"),
                "start": total,
                "end": total + num,
                "count": num,
            })
            total += num

        if not self.files:
            raise RuntimeError(f"No valid files in {chunks_dir}/{split}/")
        self.total = total

        self._cache_size = cache_size
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_order: List[str] = []

    def __len__(self) -> int:
        return self.total

    def _lookup(self, idx: int) -> Tuple[Dict, int]:
        for f in self.files:
            if idx < f["end"]:
                return f, idx - f["start"]
        raise IndexError(idx)

    def _load(self, path: str) -> torch.Tensor:
        """Return fp16 wave tensor [N, L] for this file (from LRU cache)."""
        if path in self._cache:
            return self._cache[path]["x_wave"]
        data = torch.load(path, map_location="cpu", weights_only=True)
        entry = {"x_wave": data["x_wave"]}
        if len(self._cache_order) >= self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        self._cache[path] = entry
        self._cache_order.append(path)
        return entry["x_wave"]

    def __getitem__(self, idx: int) -> Dict:
        f, chunk_idx = self._lookup(idx)
        wave = self._load(f["path"])[chunk_idx].float().clone()
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        return {"x_wave": wave, "source": f["source"]}


class FileGroupedSampler(Sampler[int]):
    """
    Shuffle tracks between epochs, then emit all chunks of each track in a
    shuffled order. Keeps per-file I/O hot, so the dataset's LRU cache of
    decoded .pt files gets hit for every chunk drawn from the same file.
    """

    def __init__(self, dataset: WaveformDataset, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.dataset.total

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        file_order = list(range(len(self.dataset.files)))
        if self.shuffle:
            file_order = torch.randperm(len(file_order), generator=g).tolist()

        for fi in file_order:
            f = self.dataset.files[fi]
            chunk_idxs = list(range(f["start"], f["end"]))
            if self.shuffle:
                perm = torch.randperm(len(chunk_idxs), generator=g).tolist()
                chunk_idxs = [chunk_idxs[i] for i in perm]
            yield from chunk_idxs


def stratified_val_indices(val_ds: "WaveformDataset", per_source: int,
                           seed: int = 0) -> Tuple[List[int], Dict[str, int]]:
    """Return a balanced index list: up to ``per_source`` chunks PER source.

    The val set is heavily source-imbalanced (fma ~80%, musdb ~3%), so a
    proportional random draw leaves musdb with a handful of samples. This
    picks an equal quota per source (seeded), then shuffles the combined
    list so sources are interleaved across batches. Returns (indices,
    per_source_counts).
    """
    import random as _random
    by_src: Dict[str, List[int]] = {}
    for f in val_ds.files:
        by_src.setdefault(f["source"], []).extend(range(f["start"], f["end"]))
    rng = _random.Random(seed)
    picked: List[int] = []
    counts: Dict[str, int] = {}
    for src in sorted(by_src):
        idxs = by_src[src][:]
        rng.shuffle(idxs)
        take = idxs[:per_source] if per_source else idxs
        picked.extend(take)
        counts[src] = len(take)
    rng.shuffle(picked)
    return picked, counts


def build_dataloaders(
    chunks_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
    cache_size: int = 8,
    val_shuffle: bool = False,
    val_seed: int = 0,
    val_per_source: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, WaveformDataset, WaveformDataset, FileGroupedSampler]:
    train_ds = WaveformDataset(chunks_dir, split="train", cache_size=cache_size)
    val_ds = WaveformDataset(chunks_dir, split="test", cache_size=cache_size)
    train_sampler = FileGroupedSampler(train_ds, shuffle=True)
    # val is stored source-contiguous (all fma, then maestro, then musdb).
    #   val_per_source: balanced quota per source (preferred for subsampling).
    #   val_shuffle:    proportional random draw (legacy; fma-dominated).
    #   neither:        deterministic full order (default; full-eval runs).
    val_sampler = None
    if val_per_source is not None:
        idxs, counts = stratified_val_indices(val_ds, val_per_source, val_seed)
        print(f"val stratified sample: {counts} (total {len(idxs)})")
        val_sampler = idxs  # explicit index list = deterministic, balanced
    elif val_shuffle:
        val_sampler = torch.utils.data.RandomSampler(
            val_ds, generator=torch.Generator().manual_seed(val_seed))

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }
    if num_workers > 0:
        common["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            common["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, sampler=train_sampler, **common)
    # For a stratified subsample, keep every picked chunk (no drop_last) so
    # the per-source quotas are exact.
    val_common = dict(common)
    if val_per_source is not None:
        val_common["drop_last"] = False
    val_loader = DataLoader(val_ds, sampler=val_sampler,
                            shuffle=False, **val_common)
    return train_loader, val_loader, train_ds, val_ds, train_sampler
