"""Convert MoisesDB into the MUSDB18-HQ layout our subtraction eval reads.

Output, one directory per track:

    <out>/<track_id>/{mixture,drums,bass,vocals,other}.wav   44.1 kHz stereo

Two decisions, both fixed by research/paper/icassp2027/HOLDOUT_PROTOCOL.md
before any audio was scored:

1. **Stem grouping is the dataset authors' own.** We use `mix_4_stems` from
   the official `moisesdb` package, which maps the 11 top-level stems onto
   MUSDB18's four: vocals, bass, drums, and everything else as "other". We do
   not invent a grouping.
2. **The mixture is the sum of those four stems**, not the mixture MoisesDB
   ships. That makes `mixture - stem == residual` exact, so the evaluation
   measures latent arithmetic and never dataset bookkeeping. Relative stem
   gains are preserved: nothing is normalized, limited, or re-leveled.

A track that is missing one of the four groups gets a silent file for it and
is kept; the activity rule at scoring time drops those units, and the counts
are reported rather than silently absorbed.

Usage:
    pip install git+https://github.com/moises-ai/moises-db.git
    python -m scripts.prepare_moisesdb --src /path/to/moisesdb_v0.1 \
        --out dataset/moisesdb_musdbform
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
TARGETS = ["drums", "bass", "vocals", "other"]


def _complete(d: Path) -> bool:
    """True if this track directory already holds all five non-empty files."""
    return all((d / f"{s}.wav").exists() and (d / f"{s}.wav").stat().st_size > 44
               for s in ["mixture"] + TARGETS)


def _load_tracks(src: Path, out: Path):
    """Yield (track_id, {group: [n, 2] float32}) using the official package.

    Tracks already converted are yielded as (id, None) so the caller can skip
    the audio work entirely: a long conversion that dies partway is resumed
    rather than restarted, and mixing several multi-stem sources is the part
    that peaks in memory.
    """
    try:
        from moisesdb.dataset import MoisesDB
        from moisesdb.defaults import mix_4_stems
    except ImportError:
        sys.exit("moisesdb package not found. Install it with:\n"
                 "  pip install git+https://github.com/moises-ai/moises-db.git")

    db = MoisesDB(data_path=str(src), sample_rate=SR)
    print(f"{len(db)} tracks in {src}")
    print("grouping (official mix_4_stems):")
    for k, v in mix_4_stems.items():
        print(f"  {k:7s} <- {', '.join(v)}")
    for track in db:
        if _complete(out / track.id):
            yield track.id, None
            continue
        yield track.id, track.mix_stems(mix_4_stems)
        gc.collect()


def _n_samples(x) -> int:
    """Length in samples, whatever the array orientation."""
    a = np.asarray(x)
    if a.ndim == 1:
        return int(a.shape[0])
    # channels-first arrays have a tiny leading axis (1 or 2)
    return int(a.shape[1] if a.shape[0] <= 2 < a.shape[1] else a.shape[0])


def _as_stereo(x: np.ndarray, n: int) -> np.ndarray:
    """[c, n] or [n, c] or [n] -> [n, 2] float32, padded/trimmed to n."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    elif a.shape[0] < a.shape[1]:      # channels-first -> channels-last
        a = a.T
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    elif a.shape[1] > 2:
        a = a[:, :2]
    if a.shape[0] < n:
        a = np.pad(a, ((0, n - a.shape[0]), (0, 0)))
    return a[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("MOISESDB_PATH"),
                    help="MoisesDB root (the dir holding the track folders)")
    ap.add_argument("--out", default="dataset/moisesdb_musdbform")
    ap.add_argument("--limit", type=int, default=None, help="first N tracks (smoke test)")
    a = ap.parse_args()
    if not a.src:
        sys.exit("--src or MOISESDB_PATH required")
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest, n_done, n_skipped = [], 0, 0
    for tid, groups in _load_tracks(src, out):
        if a.limit is not None and n_done >= a.limit:
            break
        if groups is None:                      # already on disk from an earlier run
            d = out / tid
            n = sf.info(str(d / "mixture.wav")).frames
            manifest.append({"track": tid, "samples": int(n),
                             "seconds": round(n / SR, 2), "missing_groups": [],
                             "mixture_peak": None, "resumed": True})
            n_done += 1; n_skipped += 1
            continue
        present = {g: v for g, v in groups.items() if v is not None and np.size(v)}
        if not present:
            print(f"  [skip] {tid}: no audio"); continue
        n = max(_n_samples(v) for v in present.values())

        stems, missing = {}, []
        for g in TARGETS:
            if g in present:
                stems[g] = _as_stereo(present[g], n)
            else:
                stems[g] = np.zeros((n, 2), dtype=np.float32)
                missing.append(g)

        # Mixture is DEFINED as the sum, so subtraction is exact by construction.
        mixture = sum(stems[g] for g in TARGETS)
        peak = float(np.abs(mixture).max())
        if peak == 0.0:
            print(f"  [skip] {tid}: silent mixture"); continue

        d = out / tid
        d.mkdir(exist_ok=True)
        sf.write(str(d / "mixture.wav"), mixture, SR, subtype="FLOAT")
        for g in TARGETS:
            sf.write(str(d / f"{g}.wav"), stems[g], SR, subtype="FLOAT")

        manifest.append({"track": tid, "samples": int(n),
                         "seconds": round(n / SR, 2),
                         "missing_groups": missing,
                         "mixture_peak": round(peak, 4)})
        del stems, mixture, present, groups
        gc.collect()
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done} tracks done ({n_skipped} resumed)", flush=True)

    (out / "manifest.json").write_text(json.dumps(
        {"source": str(src), "sample_rate": SR, "targets": TARGETS,
         "mixture": "sum of the four groups (exact additivity by construction)",
         "grouping": "moisesdb.defaults.mix_4_stems",
         "n_tracks": len(manifest), "tracks": manifest}, indent=2))
    miss = [m for m in manifest if m["missing_groups"]]
    print(f"\nwrote {len(manifest)} tracks to {out} ({n_skipped} already present)")
    print(f"tracks with an empty group: {len(miss)}"
          + (f" (e.g. {miss[0]['track']}: {miss[0]['missing_groups']})" if miss else ""))
    print(f"total audio: {sum(m['seconds'] for m in manifest) / 3600:.2f} h")


if __name__ == "__main__":
    main()
