"""Generate the audio-examples page assets: stem removal via latent subtraction.

Produces docs/audio/*.wav + docs/manifest.json for the GitHub Pages listening
page (docs/index.html). Run on the machine that has the checkpoints and the
MUSDB18-HQ test set:

  python -m scripts.generate_demo_audio                 # all 4 models, defaults
  python -m scripts.generate_demo_audio --seconds 5
  python -m scripts.generate_demo_audio --stems bass drums vocals

Protocol notes (kept honest on purpose):

- Reuses evaluation.compute_subtraction's own loaders and SI-SDR — the demo
  cannot drift from the paper's protocol.
- Example selection is DETERMINISTIC and MODEL-INDEPENDENT: windows are scored
  on a fixed 0.5 s grid by the energy of the stem to be removed (subject to
  mixture and residual being non-silent). No RNG, and nothing about any
  model's output influences which windows are shown.
- Clips longer than 1 s are made from consecutive 1 s chunks processed
  independently (the paper's protocol) and overlap-added with a short linear
  crossfade (XFADE samples). References are raw slices of the same window.
- All files of one example share a single gain (peak -> 0.95), so relative
  levels between mixture / stem / target / model outputs stay truthful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from evaluation.compute_subtraction import (
    CHUNK_LEN, SR, STEMS, _load_track_stems, _si_sdr_batch, _stem_sanity)
from models.autoencoder import Autoencoder
from training.config import build_model_config, get_device, load_config

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

XFADE = 1024          # samples of linear crossfade between adjacent chunks
GRID_HOP = SR // 2    # candidate-window grid: every 0.5 s

# (run name, page label) — order is display order on the page.
MODELS = [
    ("v2.0-continued",       "7.66×, no mix (v2.0)"),
    ("v2.1-decmix",          "7.66×, +ℒ_dec (v2.1)"),
    ("v3.0-baseline-d64",    "15.3×, no mix (v3.0)"),
    ("v3.1-decmix-disc-d64", "15.3×, +mix (v3.1)"),
]

# checkpoints/<run>/config.yaml is a copy written at training time. When only
# best.pth was fetched (e.g. from the HF checkpoints repo), fall back to the
# tracked experiment config; load_config() merges configs/base.yaml itself.
CONFIG_FALLBACK = {
    "v2.0-continued":       "configs/experiments/v2/v2.0_continued.yaml",
    "v2.1-decmix":          "configs/experiments/v2/v2.1_decmix.yaml",
    "v3.0-baseline-d64":    "configs/experiments/v3/v3.0_baseline_d64.yaml",
    "v3.1-decmix-disc-d64": "configs/experiments/v3/v3.1_decmix_disc_d64.yaml",
}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def clip_len(seconds: int) -> int:
    """Total samples for `seconds` consecutive chunks at hop CHUNK_LEN-XFADE."""
    n_chunks = max(1, int(seconds))
    return CHUNK_LEN + (n_chunks - 1) * (CHUNK_LEN - XFADE)


def select_examples(musdb_dir: Path, stems_wanted, seconds: int,
                    per_stem: int, sanity_thresh: float = 20.0):
    """Best demo window per (stem, track); top `per_stem` tracks per stem.

    Score = RMS of the stem to remove over the whole clip window, provided the
    mixture and the residual are both audibly non-silent. Deterministic.
    """
    total = clip_len(seconds)
    tracks = sorted(p for p in musdb_dir.iterdir() if p.is_dir())
    best = {st: [] for st in stems_wanted}   # stem -> [(score, track, start)]

    for t in tracks:
        try:
            stems = _load_track_stems(t)
        except Exception as e:
            print(f"  !! {t.name}: load failed: {e}")
            continue
        if _stem_sanity(stems) < sanity_thresh:
            print(f"  skipping {t.name}: stem identity below {sanity_thresh} dB")
            continue
        n = min(len(s) for s in stems.values())
        if n < total:
            continue
        for st in stems_wanted:
            top = None
            for s0 in range(0, n - total, GRID_HOP):
                mix = stems["mixture"][s0:s0 + total]
                stem = stems[st][s0:s0 + total]
                if _rms(mix) < 1e-3:
                    continue
                if _rms(stem) < 0.01 or _rms(mix - stem) < 0.01:
                    continue                       # inaudible stem or residual
                sc = _rms(stem)
                if top is None or sc > top[0]:
                    top = (sc, s0)
            if top is not None:
                best[st].append((top[0], t.name, top[1]))
        del stems

    examples = []
    for st in stems_wanted:
        for sc, track, s0 in sorted(best[st], reverse=True)[:per_stem]:
            examples.append({"stem": st, "track": track, "start": int(s0),
                             "stem_rms": round(sc, 4)})
    return examples


@torch.no_grad()
def render(model, x: np.ndarray, seconds: int, device, seed: int) -> np.ndarray:
    """Encode-decode x (or any latent op via caller) chunk-by-chunk with OLA."""
    raise NotImplementedError  # replaced by render_op below


@torch.no_grad()
def render_op(model, op, seconds: int, device, seed: int) -> np.ndarray:
    """Run `op(chunk_slice) -> latent` for each 1 s chunk, decode, overlap-add.

    op receives (start, end) sample indices into the clip and must return the
    latent to decode for that chunk. Chunks are processed independently, per
    the paper's protocol; adjacent outputs are crossfaded over XFADE samples.
    """
    n_chunks = max(1, int(seconds))
    hop = CHUNK_LEN - XFADE
    total = CHUNK_LEN + (n_chunks - 1) * hop
    out = np.zeros(total, dtype=np.float64)
    ramp_in = np.linspace(0.0, 1.0, XFADE)
    for i in range(n_chunks):
        pos = i * hop
        torch.manual_seed(seed)
        z = op(pos, pos + CHUNK_LEN)
        torch.manual_seed(seed)
        y, _ = model.decoder(z)
        y = y.squeeze(0).squeeze(0).float().cpu().numpy()[:CHUNK_LEN]
        if len(y) < CHUNK_LEN:                     # defensive; v2/v3 emit 44100
            y = np.pad(y, (0, CHUNK_LEN - len(y)))
        y = y.astype(np.float64)
        if i > 0:
            y[:XFADE] *= ramp_in
        if i < n_chunks - 1:
            y[-XFADE:] *= ramp_in[::-1]
        out[pos:pos + CHUNK_LEN] += y
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb-dir", type=str,
                    default=str(REPO / "dataset" / "musdb18" / "test"))
    ap.add_argument("--stems", nargs="+", default=list(STEMS),
                    choices=list(STEMS))
    ap.add_argument("--examples-per-stem", type=int, default=1)
    ap.add_argument("--seconds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-name", default="best.pth")
    ap.add_argument("--out-dir", type=str, default=str(DOCS))
    args = ap.parse_args()

    device = get_device()
    musdb = Path(args.musdb_dir)
    if not musdb.exists():
        raise SystemExit(f"musdb dir not found: {musdb}")
    out_root = Path(args.out_dir)
    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("selecting examples (deterministic, model-independent)...")
    examples = select_examples(musdb, args.stems, args.seconds,
                               args.examples_per_stem)
    if not examples:
        raise SystemExit("no examples found — check --musdb-dir")
    for ex in examples:
        print(f"  {ex['stem']:7s}  {ex['track']}  @ {ex['start']/SR:.1f}s")

    total = clip_len(args.seconds)
    manifest = {
        "sr": SR,
        "seconds": round(total / SR, 3),
        "protocol": {
            "chunk_seconds": 1.0,
            "crossfade_samples": XFADE,
            "selection": "fixed 0.5 s grid, argmax stem RMS; "
                         "independent of all model outputs",
            "shared_gain_per_example": True,
            "seed": args.seed,
            "checkpoint_name": args.checkpoint_name,
        },
        "models": [{"name": n, "label": l} for n, l in MODELS],
        "examples": [],
    }

    # References + per-model outputs, one model in memory at a time.
    clips = {}   # (ex_idx, kind) -> np.ndarray
    for i, ex in enumerate(examples):
        stems = _load_track_stems(musdb / ex["track"])
        s0 = ex["start"]
        mix = stems["mixture"][s0:s0 + total].astype(np.float32)
        stem = stems[ex["stem"]][s0:s0 + total].astype(np.float32)
        clips[(i, "mixture")] = mix
        clips[(i, "stem")] = stem
        clips[(i, "target")] = mix - stem
        del stems

    for name, label in MODELS:
        ckpt_dir = REPO / "checkpoints" / name
        cfg_path = ckpt_dir / "config.yaml"
        if not cfg_path.exists():
            cfg_path = REPO / CONFIG_FALLBACK[name]
        ckpt_path = ckpt_dir / args.checkpoint_name
        if not ckpt_path.exists():
            raise SystemExit(f"missing {ckpt_path} — download the checkpoints "
                             f"first (see README / scripts/slurm_demo_audio.sh)")
        cfg = load_config(str(cfg_path))
        model = Autoencoder(**build_model_config(cfg)).to(device)
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"loaded {name} @ step {ck.get('global_step', -1)}")

        for i, ex in enumerate(examples):
            mix = torch.from_numpy(clips[(i, "mixture")]).to(device)
            stm = torch.from_numpy(clips[(i, "stem")]).to(device)
            tgt = torch.from_numpy(clips[(i, "target")]).to(device)

            def op_sub(a, b):
                torch.manual_seed(args.seed)
                zf = model.encoder(mix[a:b].view(1, 1, -1))
                torch.manual_seed(args.seed)
                zs = model.encoder(stm[a:b].view(1, 1, -1))
                return zf - zs

            def op_ceil(a, b):
                torch.manual_seed(args.seed)
                return model.encoder(tgt[a:b].view(1, 1, -1))

            clips[(i, f"sub_{name}")] = render_op(
                model, op_sub, args.seconds, device, args.seed)
            clips[(i, f"ceil_{name}")] = render_op(
                model, op_ceil, args.seconds, device, args.seed)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # SI-SDR per clip + shared-gain write-out.
    for i, ex in enumerate(examples):
        tgt = torch.from_numpy(clips[(i, "target")])
        entry = dict(ex)
        entry["start_seconds"] = round(ex["start"] / SR, 2)
        entry["files"] = {k: f"audio/ex{i}_{k}.wav"
                          for k in ["mixture", "stem", "target"]}
        entry["metrics"] = {}
        keys = ["mixture", "stem", "target"]
        for name, _ in MODELS:
            for kind in ("sub", "ceil"):
                k = f"{kind}_{name}"
                keys.append(k)
                entry["files"][k] = f"audio/ex{i}_{k}.wav"
                y = torch.from_numpy(clips[(i, k)])
                L = min(len(y), len(tgt))
                sdr = _si_sdr_batch(y[:L].view(1, -1), tgt[:L].view(1, -1))
                entry["metrics"][k] = round(float(sdr[0]), 2)

        peak = max(float(np.abs(clips[(i, k)]).max()) for k in keys) or 1.0
        gain = min(1.0, 0.95 / peak)
        entry["gain"] = round(gain, 4)
        for k in keys:
            sf.write(str(out_root / entry["files"][k]),
                     (clips[(i, k)] * gain).astype(np.float32), SR,
                     subtype="PCM_16")
        manifest["examples"].append(entry)
        print(f"ex{i} ({ex['stem']} / {ex['track']}): " +
              "  ".join(f"{n}={entry['metrics'][f'sub_{n}']:+.1f}dB"
                        for n, _ in MODELS))

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    n_files = 3 + 2 * len(MODELS)
    print(f"\nwrote {len(examples)} examples x {n_files} wavs + manifest.json "
          f"under {out_root}")
    print("next: git add docs && commit && push, then enable GitHub Pages "
          "(Settings -> Pages -> main /docs)")


if __name__ == "__main__":
    main()
