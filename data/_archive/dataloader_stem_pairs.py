"""
ARCHIVED: Stem-pair dataloader (Option β — same-track mixing).

This was originally used by the v6 "frankenstein" phase-2 run, where mixing
happened on aligned stems from the SAME track (drums+bass, vocals+other, ...).
That path was removed when we consolidated to Option α (random cross-batch
mixing inside Autoencoder.forward) for the ISMIR paper.

Kept here in case we need to run an Option β control to test the alternative
hypothesis that same-track stem mixing was the real phase anchor in v6.
See memory: project_stem_pair_phase_hypothesis.md.
"""

import os
import json
import random
from typing import Optional, Tuple, List, Dict, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class StemPairDataset(Dataset):
    """
    Dataset that returns aligned stem pairs from preprocessed chunks.

    Each __getitem__ returns a dict with:
        x_stft:  [2, F, T]  - stem A STFT (for reconstruction loss)
        x_wave:  [1, L]     - stem A waveform
        x_stft2: [2, F, T]  - stem B STFT (paired for mixing loss)
        x_wave2: [1, L]     - stem B waveform
    """

    def __init__(
        self,
        chunks_dir: str,
        index_path: str,
        split: str,
        dtype: torch.dtype = torch.float32,
        augment: bool = False,
    ):
        self.chunks_dir = chunks_dir
        self.split = split
        self.augment = augment
        self.dtype = dtype

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"index.json not found at {index_path}. "
                "Run preprocessing first."
            )

        with open(index_path, "r") as f:
            index = json.load(f)

        if split not in index:
            raise ValueError(
                f"Split '{split}' not in index. Available: {list(index.keys())}"
            )

        self.tracks: List[Tuple[str, int, Dict[str, str]]] = []
        total_chunks = 0

        for track_key, track_info in sorted(index[split].items()):
            num_chunks = track_info["num_chunks"]
            stems = track_info["stems"]

            stem_paths = {}
            all_exist = True
            for stem_name, filename in stems.items():
                path = os.path.join(chunks_dir, split, filename)
                if not os.path.exists(path):
                    all_exist = False
                    break
                stem_paths[stem_name] = path

            if not all_exist or num_chunks == 0:
                continue

            self.tracks.append((track_key, num_chunks, stem_paths))
            total_chunks += num_chunks

        if not self.tracks:
            raise RuntimeError(f"No valid tracks found for split '{split}'.")

        self.total_chunks = total_chunks
        self.stem_names: List[str] = list(self.tracks[0][2].keys())

        self._cumulative = []
        cum = 0
        for _, num_chunks, _ in self.tracks:
            cum += num_chunks
            self._cumulative.append(cum)

        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_keys: List[str] = []
        self._max_cache = 4

    def __len__(self):
        return self.total_chunks

    def _global_to_local(self, idx: int) -> Tuple[int, int]:
        for ti, cum in enumerate(self._cumulative):
            if idx < cum:
                prev = self._cumulative[ti - 1] if ti > 0 else 0
                return ti, idx - prev
        raise IndexError(f"Index {idx} out of range for {self.total_chunks} chunks")

    def _load_file(self, path: str) -> Dict[str, torch.Tensor]:
        if path in self._cache:
            return self._cache[path]

        data = torch.load(path, map_location="cpu", weights_only=True)
        result = {"x_stft": data["x_stft"], "x_wave": data["x_wave"]}

        if len(self._cache_keys) >= self._max_cache:
            old_key = self._cache_keys.pop(0)
            self._cache.pop(old_key, None)

        self._cache[path] = result
        self._cache_keys.append(path)
        return result

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        track_idx, chunk_idx = self._global_to_local(idx)
        _, _, stem_paths = self.tracks[track_idx]

        stem_keys = list(stem_paths.keys())
        stem_a, stem_b = random.sample(stem_keys, 2)

        data_a = self._load_file(stem_paths[stem_a])
        data_b = self._load_file(stem_paths[stem_b])

        x_stft = data_a["x_stft"][chunk_idx].to(self.dtype).clone()
        x_wave = data_a["x_wave"][chunk_idx].to(self.dtype).clone()
        x_stft2 = data_b["x_stft"][chunk_idx].to(self.dtype).clone()
        x_wave2 = data_b["x_wave"][chunk_idx].to(self.dtype).clone()

        if x_wave.dim() == 1:
            x_wave = x_wave.unsqueeze(0)
        if x_wave2.dim() == 1:
            x_wave2 = x_wave2.unsqueeze(0)

        if self.augment:
            gain1 = 10 ** (torch.empty(1).uniform_(-6, 6) / 20)
            gain2 = 10 ** (torch.empty(1).uniform_(-6, 6) / 20)
            x_stft, x_wave = x_stft * gain1, x_wave * gain1
            x_stft2, x_wave2 = x_stft2 * gain2, x_wave2 * gain2

        return {
            "x_stft": x_stft,
            "x_wave": x_wave,
            "x_stft2": x_stft2,
            "x_wave2": x_wave2,
        }


class TrackSampler(Sampler[int]):
    """Shuffles at track level, keeps chunks within a track together."""

    def __init__(
        self,
        dataset: StemPairDataset,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.dataset.total_chunks

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        track_indices = list(range(len(self.dataset.tracks)))
        if self.shuffle:
            track_indices = torch.randperm(len(track_indices), generator=g).tolist()

        indices: List[int] = []
        for ti in track_indices:
            prev = self.dataset._cumulative[ti - 1] if ti > 0 else 0
            num_chunks = self.dataset.tracks[ti][1]
            chunk_indices = list(range(prev, prev + num_chunks))

            if self.shuffle:
                perm = torch.randperm(len(chunk_indices), generator=g).tolist()
                chunk_indices = [chunk_indices[i] for i in perm]

            indices.extend(chunk_indices)

        return iter(indices)


def build_stem_pair_dataloaders(
    chunks_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    logger: Optional[object] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, StemPairDataset, StemPairDataset, TrackSampler]:
    """Create train/val dataloaders for stem pairs."""
    index_path = os.path.join(chunks_dir, "index.json")

    if pin_memory is None:
        pin_memory = device.type == "cuda"
    if persistent_workers is None:
        persistent_workers = pin_memory and num_workers > 0

    train_dataset = StemPairDataset(
        chunks_dir=chunks_dir,
        index_path=index_path,
        split="train",
        augment=True,
    )
    val_dataset = StemPairDataset(
        chunks_dir=chunks_dir,
        index_path=index_path,
        split="test",
    )

    train_sampler = TrackSampler(train_dataset, shuffle=True)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    if logger:
        logger.info(
            f"Stem-pair train: {len(train_dataset):,} chunks from "
            f"{len(train_dataset.tracks)} tracks"
        )
        logger.info(
            f"Stem-pair val: {len(val_dataset):,} chunks from "
            f"{len(val_dataset.tracks)} tracks"
        )

    return train_loader, val_loader, train_dataset, val_dataset, train_sampler
