"""Chunk a held-out corpus (MUSDB18-HQ layout) into the .pt format the mixing
evaluator reads, so convex-mixing metrics run on the same audio as subtraction.

Only the mixtures are chunked: the mixing probe needs music, not stems.

    <out>/test/<track>.pt      {"x_wave": [N, L] fp16}
    <out>/index.json           {"test": {key: {filename, num_chunks, source}}}

Chunk selection is deterministic (fixed stride from the start of the track),
so every model sees identical units, and it never depends on any score.

Usage:
  python -m scripts.chunk_holdout --src dataset/moisesdb_musdbform \
      --out chunks-holdout --seconds 1.0 --max-chunks-per-track 60
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 44100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of <track>/mixture.wav")
    ap.add_argument("--out", default="chunks-holdout")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--max-chunks-per-track", type=int, default=60)
    ap.add_argument("--source", default="moisesdb", help="source tag in the index")
    ap.add_argument("--min-rms", type=float, default=1e-3,
                    help="drop near-silent chunks (same floor as the MUSDB eval)")
    a = ap.parse_args()

    L = int(round(a.seconds * SR))
    src, out = Path(a.src), Path(a.out)
    (out / "test").mkdir(parents=True, exist_ok=True)
    index = {"test": {}}
    tracks = sorted(p for p in src.iterdir() if p.is_dir() and (p / "mixture.wav").exists())
    print(f"{len(tracks)} tracks in {src}")

    total = 0
    for t in tracks:
        wav, sr = sf.read(str(t / "mixture.wav"), dtype="float32", always_2d=True)
        if sr != SR:
            print(f"  [skip] {t.name}: sr={sr}"); continue
        mono = wav.mean(axis=1)
        n_full = len(mono) // L
        if n_full == 0:
            print(f"  [skip] {t.name}: shorter than one chunk"); continue
        # Even coverage across the track, deterministic, no randomness.
        idx = np.linspace(0, n_full - 1, min(n_full, a.max_chunks_per_track))
        idx = sorted(set(int(round(i)) for i in idx))
        chunks = []
        for i in idx:
            c = mono[i * L:(i + 1) * L]
            if float(np.sqrt((c.astype(np.float64) ** 2).mean())) < a.min_rms:
                continue
            chunks.append(c)
        if not chunks:
            print(f"  [skip] {t.name}: all chunks below the silence floor"); continue
        x = torch.from_numpy(np.stack(chunks)).half()
        key = f"{a.source}__{t.name}"
        fn = f"{key}.pt"
        torch.save({"x_wave": x}, out / "test" / fn)
        index["test"][key] = {"filename": fn, "num_chunks": int(x.size(0)),
                              "source": a.source}
        total += int(x.size(0))

    (out / "index.json").write_text(json.dumps(index))
    print(f"wrote {len(index['test'])} files, {total:,} chunks of {a.seconds}s to {out}")


if __name__ == "__main__":
    main()
