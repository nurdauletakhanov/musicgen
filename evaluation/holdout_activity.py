"""Target-activity measurements for the pre-registered silent-target rule.

HOLDOUT_PROTOCOL.md section 4 fixes two thresholds before any scoring: a unit
counts only if the target stem in that chunk has RMS >= -50 dBFS and a
target-to-mixture energy ratio >= -30 dB. SI-SDR against a near-silent
reference is unstable, and near-silent targets are the mechanism behind the
extreme per-unit scores in the MUSDB evaluation (values from -78 to +42 dB).

The subtraction evaluator does not record those quantities, but every per-unit
record carries the exact sample offset of its chunk, so they are recovered
here from the audio alone. No model, no GPU, no re-run: the same units, read
straight from the corpus.

Usage:
  python -m evaluation.holdout_activity \
      --per-chunk evaluation/holdout_moisesdb/per_chunk/v2.0-continued_per_chunk.json \
      --corpus dataset/moisesdb_musdbform \
      --out evaluation/holdout_moisesdb/activity.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

SR = 44100
CHUNK_LEN = SR  # 1 s, matching evaluation/compute_subtraction.py
EPS = 1e-12


def _read(path, start, n):
    x, sr = sf.read(path, start=start, frames=n, dtype="float32", always_2d=True)
    if sr != SR:
        raise RuntimeError(f"unexpected sr={sr} in {path}")
    return x.mean(axis=1)  # the evaluator scores mono


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-chunk", required=True,
                    help="any one model's per-unit file; units are identical across models")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = json.load(open(a.per_chunk))["records"]
    by_track = defaultdict(list)
    for r in recs:
        by_track[r["track"]].append(r)
    print(f"{len(recs)} units across {len(by_track)} tracks")

    out = {}
    for ti, (track, rows) in enumerate(sorted(by_track.items()), 1):
        d = os.path.join(a.corpus, track)
        mix_path = os.path.join(d, "mixture.wav")
        for r in rows:
            start = int(r["start"])
            stem = r["stem"]
            tgt = _read(os.path.join(d, f"{stem}.wav"), start, CHUNK_LEN)
            mix = _read(mix_path, start, CHUNK_LEN)
            t_rms = float(np.sqrt(np.mean(tgt.astype(np.float64) ** 2)))
            t_e = float(np.sum(tgt.astype(np.float64) ** 2))
            m_e = float(np.sum(mix.astype(np.float64) ** 2))
            out[f"{track}|{stem}|{r['chunk']}"] = {
                "rms_dbfs": 20.0 * np.log10(max(t_rms, EPS)),
                "target_to_mix_db": 10.0 * np.log10(max(t_e, EPS) / max(m_e, EPS)),
            }
        if ti % 40 == 0:
            print(f"  {ti} tracks", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))

    rms = np.array([v["rms_dbfs"] for v in out.values()])
    rat = np.array([v["target_to_mix_db"] for v in out.values()])
    keep = (rms >= -50.0) & (rat >= -30.0)
    print(f"\nwrote {a.out}  ({len(out)} units)")
    print(f"pre-registered rule (RMS >= -50 dBFS and target/mix >= -30 dB):")
    print(f"  kept    {int(keep.sum()):6d} ({100*keep.mean():.1f}%)")
    print(f"  dropped {int((~keep).sum()):6d} "
          f"({int((rms < -50).sum())} by RMS, {int((rat < -30).sum())} by ratio)")
    per_stem = defaultdict(lambda: [0, 0])
    for k, v in out.items():
        st = k.split("|")[1]
        per_stem[st][1] += 1
        if v["rms_dbfs"] >= -50.0 and v["target_to_mix_db"] >= -30.0:
            per_stem[st][0] += 1
    for st, (k, n) in sorted(per_stem.items()):
        print(f"  {st:8s} kept {k}/{n} ({100*k/n:.1f}%)")


if __name__ == "__main__":
    main()
