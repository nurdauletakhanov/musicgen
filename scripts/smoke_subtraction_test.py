"""
Phase A smoke test for latent subtraction on MUSDB18-HQ.

For each (model, track, chunk, stem):
  z_full   = f(x_full)
  z_stem   = f(x_stem)
  x_hat    = g(z_full - z_stem)
  target   = x_full - x_stem
  SI-SDR(x_hat, target)

Plus the encode-decode ceiling SI-SDR(g(f(target)), target) for comparison.

Sanity: SI-SDR(target, x_drums + x_bass + x_vocals + x_other - x_stem)
should be very high (MUSDB stems sum to mixture exactly).

Run:
  python -m scripts.smoke_subtraction_test
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from models.autoencoder import Autoencoder
from training.config import build_model_config, get_device, load_config

REPO = Path(__file__).resolve().parents[1]
SMOKE_DIR = REPO / "musdb_smoke" / "test"

MODELS = [
    ("v2.0-continued",  REPO / "checkpoints" / "v2.0-continued"),
    ("v2.2-decmix-disc", REPO / "checkpoints" / "v2.2-decmix-disc"),
]
STEMS = ["drums", "bass", "vocals", "other"]


def si_sdr(x_hat: np.ndarray, x: np.ndarray, eps: float = 1e-8) -> float:
    x = x.astype(np.float64) - x.astype(np.float64).mean()
    x_hat = x_hat.astype(np.float64) - x_hat.astype(np.float64).mean()
    alpha = (x_hat * x).sum() / (np.square(x).sum() + eps)
    s_target = alpha * x
    e_noise = x_hat - s_target
    return float(10.0 * np.log10(
        (np.square(s_target).sum() + eps) / (np.square(e_noise).sum() + eps)
    ))


def load_stem_mono(path: Path) -> np.ndarray:
    """Load .wav, convert to mono float32 at native SR (verify 44.1k)."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == 44100, f"unexpected sr {sr} in {path}"
    return wav.mean(axis=1)  # [T]


def load_track_stems(track_dir: Path) -> dict:
    out = {}
    for stem in ["mixture"] + STEMS:
        out[stem] = load_stem_mono(track_dir / f"{stem}.wav")
    # Sanity: stems should sum to mixture exactly. Verify identity holds.
    summed = out["drums"] + out["bass"] + out["vocals"] + out["other"]
    n = min(len(summed), len(out["mixture"]))
    sanity_sdr = si_sdr(summed[:n], out["mixture"][:n])
    return out, sanity_sdr


def pick_chunks(stems: dict, n_chunks: int, chunk_len: int, rng: random.Random,
                rms_thresh: float = 1e-3) -> list:
    """Random non-silent chunk start indices, aligned across stems."""
    n = min(len(s) for s in stems.values())
    starts = []
    tries = 0
    while len(starts) < n_chunks and tries < n_chunks * 30:
        s = rng.randrange(0, n - chunk_len)
        seg = stems["mixture"][s:s + chunk_len]
        if np.sqrt(np.mean(seg ** 2)) >= rms_thresh:
            starts.append(s)
        tries += 1
    return starts


@torch.no_grad()
def model_run(model: Autoencoder, x: torch.Tensor) -> torch.Tensor:
    """Encode-then-decode. x: [B, T] -> [B, T]."""
    z = model.encoder(x.unsqueeze(1))
    y, _ = model.decoder(z)
    return y.squeeze(1)


@torch.no_grad()
def latent_subtract(model: Autoencoder, x_full: torch.Tensor,
                    x_stem: torch.Tensor) -> torch.Tensor:
    """g(f(x_full) - f(x_stem))."""
    z_full = model.encoder(x_full.unsqueeze(1))
    z_stem = model.encoder(x_stem.unsqueeze(1))
    z_diff = z_full - z_stem
    y, _ = model.decoder(z_diff)
    return y.squeeze(1)


def load_model(ckpt_dir: Path, device: torch.device) -> Autoencoder:
    cfg = load_config(str(ckpt_dir / "config.yaml"))
    model = Autoencoder(**build_model_config(cfg)).to(device)
    ck = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chunks", type=int, default=10,
                    help="Random non-silent chunks per track")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-wavs", action="store_true",
                    help="Save 1-2 example outputs as .wav for spot-check")
    args = ap.parse_args()

    device = get_device()
    rng = random.Random(args.seed)
    chunk_len = 44100  # 1 second @ 44.1 kHz

    tracks = sorted([p for p in SMOKE_DIR.iterdir() if p.is_dir()])
    if not tracks:
        raise SystemExit(f"no tracks under {SMOKE_DIR}")
    print(f"smoke set: {len(tracks)} tracks under {SMOKE_DIR}")
    for t in tracks:
        print(f"  - {t.name}")

    print("\nloading & sanity-checking stems...")
    track_data = {}
    for t in tracks:
        stems, sanity = load_track_stems(t)
        print(f"  {t.name:60s}  sum-of-stems vs mixture SI-SDR = {sanity:+.2f} dB")
        if sanity < 60:
            print(f"    !! stem identity weak — possible loading issue")
        track_data[t.name] = stems

    # Pick chunks per track (same chunks used across models for fair comparison)
    chunk_idx = {
        name: pick_chunks(stems, args.n_chunks, chunk_len, rng)
        for name, stems in track_data.items()
    }
    total_chunks = sum(len(v) for v in chunk_idx.values())
    print(f"\nselected {total_chunks} non-silent 1s chunks total")

    # Build chunk tensors once: shape [N, T] for mixture and each stem
    # Order: track1_c1, track1_c2, ..., track2_c1, ...
    mix_arr, stem_arrs = [], {s: [] for s in STEMS}
    for name in chunk_idx:
        td = track_data[name]
        for s in chunk_idx[name]:
            mix_arr.append(td["mixture"][s:s + chunk_len])
            for stem in STEMS:
                stem_arrs[stem].append(td[stem][s:s + chunk_len])
    x_mix = torch.from_numpy(np.stack(mix_arr)).to(device)         # [N, T]
    x_stems = {
        stem: torch.from_numpy(np.stack(stem_arrs[stem])).to(device)
        for stem in STEMS
    }
    targets = {stem: (x_mix - x_stems[stem]) for stem in STEMS}    # [N, T] each
    print(f"chunk tensor: {tuple(x_mix.shape)}, dtype {x_mix.dtype}")

    # Process each model
    print("\n" + "=" * 78)
    summary = {}    # {model_name: {stem: {"sub": mean_sdr, "ceil": mean_sdr}}}
    for mname, mdir in MODELS:
        print(f"\n=== {mname} ===")
        model = load_model(mdir, device)

        per_stem = {}
        # Process in mini-batches to avoid OOM
        batch = 16
        for stem in STEMS:
            sub_sdrs, ceil_sdrs = [], []
            for i in range(0, x_mix.size(0), batch):
                xm = x_mix[i:i + batch]
                xs = x_stems[stem][i:i + batch]
                tg = targets[stem][i:i + batch]
                # Latent subtraction
                x_hat = latent_subtract(model, xm, xs)
                # Encode-decode ceiling on the residual itself
                x_ceil = model_run(model, tg)
                # Crop to match (decoder output may be slightly shorter)
                L = min(x_hat.size(1), tg.size(1), x_ceil.size(1))
                for b in range(xm.size(0)):
                    sub_sdrs.append(si_sdr(
                        x_hat[b, :L].cpu().numpy(), tg[b, :L].cpu().numpy()))
                    ceil_sdrs.append(si_sdr(
                        x_ceil[b, :L].cpu().numpy(), tg[b, :L].cpu().numpy()))
            sub_mean = float(np.mean(sub_sdrs))
            ceil_mean = float(np.mean(ceil_sdrs))
            per_stem[stem] = {"sub": sub_mean, "ceil": ceil_mean}
            print(f"  remove {stem:7s}  sub={sub_mean:+6.2f} dB"
                  f"   ceil={ceil_mean:+6.2f} dB"
                  f"   gap={ceil_mean - sub_mean:+5.2f}")

        summary[mname] = per_stem

        if args.render_wavs:
            # Render a single 6 s clip (6 contiguous chunks from track 0) so the
            # listening test compares meaningful musical material, not 1 s blips.
            n_clip = 6
            clip_idx = list(range(min(n_clip, x_mix.size(0))))
            with torch.no_grad():
                xm_clip = x_mix[clip_idx]                        # [n, T]
                # encode-decode ceiling on full mix (for reference)
                x_mix_recon = model_run(model, xm_clip)
                # for each stem: latent-subtract + ceiling on the residual
                outs = {}
                for stem in STEMS:
                    xs_clip = x_stems[stem][clip_idx]
                    tg_clip = xm_clip - xs_clip
                    x_sub = latent_subtract(model, xm_clip, xs_clip)
                    x_ceil = model_run(model, tg_clip)
                    L = min(x_sub.size(1), tg_clip.size(1), x_ceil.size(1))
                    outs[stem] = {
                        "target": tg_clip[:, :L].cpu().numpy(),
                        "sub":    x_sub[:, :L].cpu().numpy(),
                        "ceil":   x_ceil[:, :L].cpu().numpy(),
                    }

            out_dir = REPO / "evaluation" / "v2_metrics" / "smoke_audio"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Concatenate the clip's chunks back into one continuous waveform per stem
            def cat(arr):
                return np.concatenate([arr[i] for i in range(arr.shape[0])])
            # Write the input mixture once (model-independent — write under v2.0 only)
            if mname == MODELS[0][0]:
                sf.write(out_dir / "00_input_mixture.wav",
                         cat(xm_clip.cpu().numpy()), 44100)
                sf.write(out_dir / "00_input_mixture_recon_v2.0.wav",
                         cat(x_mix_recon.cpu().numpy()), 44100)
                for stem in STEMS:
                    sf.write(out_dir / f"01_target_remove_{stem}.wav",
                             cat(outs[stem]["target"]), 44100)
            else:
                sf.write(out_dir / f"00_input_mixture_recon_{mname}.wav",
                         cat(x_mix_recon.cpu().numpy()), 44100)

            for stem in STEMS:
                sf.write(out_dir / f"sub_{mname}_remove_{stem}.wav",
                         cat(outs[stem]["sub"]), 44100)
                sf.write(out_dir / f"ceil_{mname}_remove_{stem}.wav",
                         cat(outs[stem]["ceil"]), 44100)
            print(f"  wrote example wavs to {out_dir}")

        del model
        torch.cuda.empty_cache()

    # Final comparison table
    print("\n" + "=" * 78)
    print("\nfinal: per-stem latent-subtraction SI-SDR (dB), higher is better")
    print(f"{'stem':9s}  " + "  ".join(f"{m:>22s}" for m, _ in MODELS) + "   delta v2.2-v2.0")
    for stem in STEMS:
        row = f"{stem:9s}  "
        v20 = summary["v2.0-continued"][stem]["sub"]
        v22 = summary["v2.2-decmix-disc"][stem]["sub"]
        for m, _ in MODELS:
            entry = summary[m][stem]
            row += f"  sub={entry['sub']:+6.2f} ceil={entry['ceil']:+6.2f}"
        row += f"      d_sub={v22 - v20:+5.2f}"
        print(row)


if __name__ == "__main__":
    main()
