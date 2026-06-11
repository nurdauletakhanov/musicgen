"""
Unified preprocessor for v1: writes a single chunks directory shared across
MUSDB, MAESTRO, and FMA.

Output layout:
  <out>/
    index.json           { "train": {key: entry}, "test": {key: entry} }
    train/<key>.pt       {"x_wave": fp16 [N, L], "peak": fp32 [N]}
    test/<key>.pt

Entry: {"filename", "num_chunks", "source"}.

Normalization: each chunk is peak-normalized to 0.95 of full scale. Original
peak is stored so absolute loudness can be recovered at inference.

Running the three sources in any order against the same output dir accumulates
into the same index; each source uses a distinct key prefix so collisions
across sources are impossible.

  python -m data.preprocess musdb   --src ./dataset/musdb18        --out ./chunks-44k-1s
  python -m data.preprocess maestro --src ./dataset/maestro-v3.0.0 --out ./chunks-44k-1s
  python -m data.preprocess fma     --src ./dataset/fma_large      --out ./chunks-44k-1s
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly
from tqdm import tqdm


# --- shared helpers ---------------------------------------------------------

def _ensure_ffmpeg_on_path():
    env_dir = os.path.dirname(sys.executable)
    lib_bin = os.path.join(env_dir, "Library", "bin")
    if os.path.isdir(lib_bin) and lib_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")


def _make_hash(s: str) -> str:
    return hashlib.md5(s.replace("\\", "/").encode("utf-8")).hexdigest()[:8]


def _load_mono(path: str, target_sr: int) -> np.ndarray:
    """Load audio as mono float32 at target_sr via libsndfile (soundfile).

    libsndfile 1.2+ has native MP3 support, so soundfile reads WAV / FLAC / OGG
    / MP3 / AIFF directly without shelling out to ffmpeg. This matters on
    Windows machines where AppControl blocks ffmpeg's DLL initialization
    (STATUS_DLL_INIT_FAILED on `ffmpeg -version`).

    Bypasses librosa.load(), so no deprecation warning from the
    audioread-fallback path scheduled for removal in librosa 1.0.
    """
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        from math import gcd
        g = gcd(int(sr), int(target_sr))
        audio = resample_poly(audio, int(target_sr) // g, int(sr) // g)

    return audio.astype(np.float32)


def _chunk_and_normalize(
    audio: np.ndarray,
    chunk_samples: int,
    min_peak: float = 0.01,
    target_peak: float = 0.95,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Split into non-overlapping chunks, peak-normalize each to `target_peak`,
    drop near-silent ones. Returns (wave[N, L] fp16, peak[N] fp32) or (None, None).
    """
    waves, peaks = [], []
    n = len(audio)
    for start in range(0, n - chunk_samples + 1, chunk_samples):
        chunk = torch.from_numpy(audio[start:start + chunk_samples]).float()
        if not torch.isfinite(chunk).all():
            continue
        peak = float(chunk.abs().max().item())
        if peak < min_peak:
            continue
        scaled = chunk * (target_peak / peak)
        waves.append(scaled)
        peaks.append(peak)
    if not waves:
        return None, None
    return (
        torch.stack(waves, dim=0).half(),
        torch.tensor(peaks, dtype=torch.float32),
    )


def _ensure_dirs(out: str):
    for s in ("train", "test"):
        os.makedirs(os.path.join(out, s), exist_ok=True)


def _load_index(out: str) -> Dict:
    path = os.path.join(out, "index.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            index = json.load(f)
        index.setdefault("train", {})
        index.setdefault("test", {})
        return index
    return {"train": {}, "test": {}}


def _save_index(out: str, index: Dict):
    path = os.path.join(out, "index.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, path)


def _save_chunks(out: str, split: str, filename: str, wave: torch.Tensor, peak: torch.Tensor):
    save_path = os.path.join(out, split, filename)
    tmp = save_path + ".tmp"
    try:
        torch.save({"x_wave": wave, "peak": peak}, tmp)
        if os.path.exists(save_path):
            os.remove(save_path)
        os.rename(tmp, save_path)
    except (RuntimeError, OSError, IOError):
        free = shutil.disk_usage(out).free / (1024 ** 3)
        print(f"\nFAILED saving {save_path}; disk free: {free:.1f} GB", file=sys.stderr)
        raise


# --- MUSDB ------------------------------------------------------------------

def preprocess_musdb(src: str, out: str, chunk_seconds: float, sample_rate: int,
                     min_peak: float, force: bool):
    """MUSDB18-HQ layout:
        <src>/{train,test}/<TrackName>/{mixture,drums,bass,other,vocals}.wav

    v1 only reads mixture.wav. The stems stay on disk for v2+ mixing experiments.
    """
    _ensure_dirs(out)
    index = _load_index(out)
    chunk_samples = int(chunk_seconds * sample_rate)

    totals = {"train": 0, "test": 0}
    skipped: List[Dict] = []

    for split in ("train", "test"):
        split_dir = os.path.join(src, split)
        if not os.path.isdir(split_dir):
            print(f"[musdb] skipping missing split: {split_dir}")
            continue

        track_dirs = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        print(f"[musdb/{split}] {len(track_dirs)} tracks")

        for track_name in tqdm(track_dirs, desc=f"musdb/{split}"):
            track_dir = os.path.join(split_dir, track_name)
            mixture_path = os.path.join(track_dir, "mixture.wav")
            if not os.path.isfile(mixture_path):
                skipped.append({"track": track_name, "reason": "no mixture.wav"})
                continue

            key = f"musdb__{track_name}__{_make_hash(os.path.join(split, track_name))}"
            filename = f"{key}.pt"

            if not force and key in index[split]:
                totals[split] += index[split][key]["num_chunks"]
                continue

            try:
                audio = _load_mono(mixture_path, sample_rate)
            except Exception as e:
                skipped.append({"track": track_name, "reason": f"decode: {e}"})
                continue

            wave, peak = _chunk_and_normalize(audio, chunk_samples, min_peak=min_peak)
            if wave is None:
                skipped.append({"track": track_name, "reason": "no valid chunks"})
                continue

            _save_chunks(out, split, filename, wave, peak)
            index[split][key] = {
                "filename": filename,
                "num_chunks": int(wave.size(0)),
                "source": "musdb",
            }
            totals[split] += int(wave.size(0))

            if (totals["train"] + totals["test"]) % 5 == 0:
                _save_index(out, index)

    _save_index(out, index)
    if skipped:
        with open(os.path.join(out, "musdb_skipped.json"), "w") as f:
            json.dump(skipped, f, indent=2)
    print(f"[musdb] done. train={totals['train']:,} chunks, test={totals['test']:,} chunks, "
          f"skipped={len(skipped)}")


# --- MAESTRO ----------------------------------------------------------------

def preprocess_maestro(src: str, out: str, chunk_seconds: float, sample_rate: int,
                       min_peak: float, force: bool):
    _ensure_dirs(out)
    index = _load_index(out)
    chunk_samples = int(chunk_seconds * sample_rate)

    csv_path = os.path.join(src, "maestro-v3.0.0.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    # Maestro splits:
    #   train, validation -> our "train" (all non-test goes into training pool)
    #   test              -> our "test"  (goes into eval pool, tagged maestro)
    rows: List[Tuple[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ours = "test" if row["split"] == "test" else "train"
            rows.append((row["audio_filename"], ours))

    n_train = sum(1 for _, s in rows if s == "train")
    n_test = sum(1 for _, s in rows if s == "test")
    print(f"[maestro] {len(rows)} tracks -> train={n_train}, test={n_test}")

    totals = {"train": 0, "test": 0}
    skipped: List[Dict] = []

    for audio_rel, our_split in tqdm(rows, desc="maestro"):
        stem = Path(audio_rel).stem
        key = f"maestro__{stem}__{_make_hash(audio_rel)}"
        filename = f"{key}.pt"

        existing = None
        for s in ("train", "test"):
            if key in index[s]:
                existing = s
                break
        if not force and existing is not None:
            totals[existing] += index[existing][key]["num_chunks"]
            continue

        audio_path = os.path.join(src, audio_rel)
        if not os.path.isfile(audio_path):
            skipped.append({"track": audio_rel, "reason": "missing"})
            continue

        try:
            audio = _load_mono(audio_path, sample_rate)
        except Exception as e:
            skipped.append({"track": audio_rel, "reason": f"decode: {e}"})
            continue

        wave, peak = _chunk_and_normalize(audio, chunk_samples, min_peak=min_peak)
        if wave is None:
            skipped.append({"track": audio_rel, "reason": "no valid chunks"})
            continue

        _save_chunks(out, our_split, filename, wave, peak)
        index[our_split][key] = {
            "filename": filename,
            "num_chunks": int(wave.size(0)),
            "source": "maestro",
        }
        totals[our_split] += int(wave.size(0))

        if (totals["train"] + totals["test"]) % 50 == 0:
            _save_index(out, index)

    _save_index(out, index)
    if skipped:
        with open(os.path.join(out, "maestro_skipped.json"), "w") as f:
            json.dump(skipped, f, indent=2)
    print(f"[maestro] done. train={totals['train']:,} chunks, test={totals['test']:,} chunks, "
          f"skipped={len(skipped)}")


# --- FMA --------------------------------------------------------------------

_FMA_WORKER: Dict = {}


def _fma_init_worker(
    out: str, chunk_samples: int, sample_rate: int, min_peak: float, force: bool,
):
    _ensure_ffmpeg_on_path()
    _FMA_WORKER["out"] = out
    _FMA_WORKER["chunk_samples"] = chunk_samples
    _FMA_WORKER["sample_rate"] = sample_rate
    _FMA_WORKER["min_peak"] = min_peak
    _FMA_WORKER["force"] = force


def _fma_process_one(mp3_path: str) -> Tuple[str, Optional[str], Optional[int], Optional[str]]:
    out = _FMA_WORKER["out"]
    chunk_samples = int(_FMA_WORKER["chunk_samples"])
    sample_rate = int(_FMA_WORKER["sample_rate"])
    min_peak = float(_FMA_WORKER["min_peak"])
    force = bool(_FMA_WORKER["force"])

    track_id = Path(mp3_path).stem  # "000002" etc.
    key = f"fma__{track_id}"
    filename = f"{key}.pt"
    save_path = os.path.join(out, "train", filename)

    if not force and os.path.exists(save_path):
        try:
            data = torch.load(save_path, map_location="cpu", weights_only=True)
            return (key, filename, int(data["x_wave"].size(0)), None)
        except Exception:
            try:
                os.remove(save_path)
            except OSError:
                pass

    try:
        audio = _load_mono(mp3_path, sample_rate)
    except Exception as e:
        return (key, None, None, f"decode: {type(e).__name__}: {e}")

    if len(audio) < 2 * sample_rate:
        return (key, None, None, f"too short: {len(audio)} samples")

    wave, peak = _chunk_and_normalize(audio, chunk_samples, min_peak=min_peak)
    if wave is None:
        return (key, None, None, "no valid chunks")

    try:
        _save_chunks(out, "train", filename, wave, peak)
    except Exception as e:
        return (key, None, None, f"save: {type(e).__name__}: {e}")

    return (key, filename, int(wave.size(0)), None)


def preprocess_fma(src: str, out: str, chunk_seconds: float, sample_rate: int,
                   min_peak: float, workers: int, force: bool, limit: Optional[int]):
    _ensure_dirs(out)
    index = _load_index(out)
    chunk_samples = int(chunk_seconds * sample_rate)

    mp3s: List[str] = []
    for sub in sorted(os.listdir(src)):
        sub_path = os.path.join(src, sub)
        if not os.path.isdir(sub_path):
            continue
        for name in sorted(os.listdir(sub_path)):
            if name.lower().endswith(".mp3"):
                mp3s.append(os.path.join(sub_path, name))

    if limit is not None:
        mp3s = mp3s[:limit]

    print(f"[fma] {len(mp3s):,} MP3s; workers={workers}")
    if not mp3s:
        return

    skipped: List[Dict] = []
    total = 0
    indexed = 0

    from multiprocessing import Pool

    with Pool(
        processes=workers,
        initializer=_fma_init_worker,
        initargs=(out, chunk_samples, sample_rate, min_peak, force),
    ) as pool:
        for key, filename, num, err in tqdm(
            pool.imap_unordered(_fma_process_one, mp3s, chunksize=8),
            total=len(mp3s),
            desc="fma",
        ):
            if err is not None:
                skipped.append({"track": key, "reason": err})
                continue
            if filename is None or num is None or num == 0:
                continue
            index["train"][key] = {
                "filename": filename,
                "num_chunks": num,
                "source": "fma",
            }
            total += num
            indexed += 1
            if indexed % 500 == 0:
                _save_index(out, index)

    _save_index(out, index)
    if skipped:
        with open(os.path.join(out, "fma_skipped.json"), "w") as f:
            json.dump(skipped, f, indent=2)
    free_gb = shutil.disk_usage(out).free / (1024 ** 3)
    print(f"[fma] done. {indexed} tracks, {total:,} chunks, "
          f"skipped {len(skipped)}. Disk free: {free_gb:.1f} GB")


# --- CLI --------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="./chunks-44k-1s")
    p.add_argument("--chunk-seconds", type=float, default=1.0)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--min-peak", type=float, default=0.01)
    p.add_argument("--force", action="store_true")
    sub = p.add_subparsers(dest="source", required=True)

    sp_musdb = sub.add_parser("musdb")
    sp_musdb.add_argument("--src", type=str, default="./dataset/musdb18")

    sp_mae = sub.add_parser("maestro")
    sp_mae.add_argument("--src", type=str, default="./dataset/maestro-v3.0.0")

    sp_fma = sub.add_parser("fma")
    sp_fma.add_argument("--src", type=str, default="./dataset/fma_large")
    sp_fma.add_argument("--workers", type=int, default=8)
    sp_fma.add_argument("--limit", type=int, default=None,
                        help="Process only the first N MP3s (smoke test)")

    args = p.parse_args()

    common = dict(
        src=args.src,
        out=args.out,
        chunk_seconds=args.chunk_seconds,
        sample_rate=args.sample_rate,
        min_peak=args.min_peak,
        force=args.force,
    )

    if args.source == "musdb":
        preprocess_musdb(**common)
    elif args.source == "maestro":
        preprocess_maestro(**common)
    elif args.source == "fma":
        preprocess_fma(workers=args.workers, limit=args.limit, **common)
    else:
        raise ValueError(args.source)


if __name__ == "__main__":
    main()
