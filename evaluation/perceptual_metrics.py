"""
Perceptual evaluation metrics: FAD and SI-SDR.

Evaluates reconstruction quality and mixing quality using automated
perceptual metrics. No human listeners needed.

Usage:
    # Evaluate our model (Phase 2, epoch 104)
    python -m evaluation.perceptual_metrics \
        --checkpoint checkpoints/musdb-phase2-mixing/checkpoint_104.pth

    # Compare Phase 1 vs Phase 2
    python -m evaluation.perceptual_metrics \
        --checkpoint checkpoints/musdb-phase2-mixing/checkpoint_104.pth \
        --checkpoint2 checkpoints/musdb-phase1-recon/best_model.pth

    # Evaluate baselines too
    python -m evaluation.perceptual_metrics \
        --checkpoint checkpoints/musdb-phase2-mixing/checkpoint_104.pth \
        --checkpoint2 checkpoints/musdb-phase1-recon/best_model.pth \
        --eval-baselines

Dependencies:
    pip install frechet-audio-distance
"""

import argparse
import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.amp import autocast

from models.autoencoder import Autoencoder


SAMPLE_RATE = 44100


def load_model(checkpoint_path: str, device: torch.device) -> Autoencoder:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Autoencoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def load_test_singles(chunks_dir: str, num_samples: int = 200) -> List[Dict]:
    """Load individual stem chunks from MUSDB test split for reconstruction eval."""
    index_path = os.path.join(chunks_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    test_tracks = index.get("test", {})
    if not test_tracks:
        raise RuntimeError("No test data found in index.json")

    stem_order = ["drums", "bass", "other", "vocals"]
    samples = []

    for track_key, track_info in test_tracks.items():
        if not isinstance(track_info, dict) or "stems" not in track_info:
            continue

        stems = track_info["stems"]
        n_chunks = track_info.get("num_chunks", 0)

        for stem_name in stem_order:
            if stem_name not in stems:
                continue
            path = os.path.join(chunks_dir, "test", stems[stem_name])
            if not os.path.exists(path):
                continue
            data = torch.load(path, map_location="cpu", weights_only=True)

            for chunk_idx in range(min(n_chunks, 5)):
                samples.append({
                    "x_stft": data["x_stft"][chunk_idx].float(),
                    "x_wave": data["x_wave"][chunk_idx].float(),
                    "track": track_key,
                    "stem": stem_name,
                })

    import random
    random.seed(42)
    random.shuffle(samples)
    return samples[:num_samples]


def load_test_stem_pairs(chunks_dir: str, num_samples: int = 200) -> List[Dict]:
    """Load stem pairs from MUSDB test split (reuse from test_evaluation)."""
    from evaluation.test_evaluation import load_test_stem_pairs as _load
    return _load(chunks_dir, num_samples)


# ---- SI-SDR ----

def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Args:
        estimate: reconstructed signal [L]
        reference: original signal [L]

    Returns:
        SI-SDR value in dB (higher is better)
    """
    # Ensure same length
    min_len = min(estimate.shape[-1], reference.shape[-1])
    estimate = estimate[..., :min_len].float()
    reference = reference[..., :min_len].float()

    # Remove mean
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()

    # s_target = <s', s> / <s, s> * s
    dot = torch.sum(estimate * reference)
    s_target = dot * reference / (torch.sum(reference ** 2) + 1e-8)

    # e_noise = s' - s_target
    e_noise = estimate - s_target

    si_sdr_val = 10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8
    )
    return si_sdr_val.item()


# ---- Reconstruction evaluation ----

def evaluate_reconstruction(
    model: torch.nn.Module,
    samples: List[Dict],
    device: torch.device,
    save_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate reconstruction quality: SI-SDR for each sample.

    If save_dir is provided, saves original and reconstructed audio files
    for FAD computation.
    """
    si_sdr_list = []
    l1_list = []

    orig_dir = None
    recon_dir = None
    if save_dir:
        orig_dir = os.path.join(save_dir, "original")
        recon_dir = os.path.join(save_dir, "reconstructed")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            x_stft = sample["x_stft"].unsqueeze(0).to(device)
            x_wave = sample["x_wave"]

            if x_wave.dim() == 1:
                x_wave_3d = x_wave.unsqueeze(0).unsqueeze(0).to(device)
            else:
                x_wave_3d = x_wave.unsqueeze(0).to(device)
                if x_wave_3d.dim() == 2:
                    x_wave_3d = x_wave_3d.unsqueeze(1)

            with autocast("cuda", enabled=True):
                z = model.encoder(x_stft)
                x_hat = model.decoder(z)

            # Trim to same length
            min_len = min(x_hat.shape[-1], x_wave_3d.shape[-1])
            x_hat = x_hat[..., :min_len]
            x_wave_trimmed = x_wave_3d[..., :min_len]

            # SI-SDR
            sdr_val = si_sdr(
                x_hat[0, 0].cpu(),
                x_wave_trimmed[0, 0].cpu(),
            )
            si_sdr_list.append(sdr_val)

            # L1
            l1_val = F.l1_loss(x_hat, x_wave_trimmed).item()
            l1_list.append(l1_val)

            # Save audio files for FAD
            if save_dir:
                orig_audio = x_wave_trimmed[0, 0].cpu().numpy()
                recon_audio = x_hat[0, 0].cpu().float().numpy()

                sf.write(
                    os.path.join(orig_dir, f"sample_{i:04d}.wav"),
                    orig_audio, SAMPLE_RATE, subtype="PCM_16",
                )
                sf.write(
                    os.path.join(recon_dir, f"sample_{i:04d}.wav"),
                    recon_audio, SAMPLE_RATE, subtype="PCM_16",
                )

            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(samples)} samples")

    return {
        "SI-SDR_mean": float(np.mean(si_sdr_list)),
        "SI-SDR_std": float(np.std(si_sdr_list)),
        "SI-SDR_median": float(np.median(si_sdr_list)),
        "L1_mean": float(np.mean(l1_list)),
        "L1_std": float(np.std(l1_list)),
        "num_samples": len(samples),
    }


def compute_fad(orig_dir: str, recon_dir: str, model_name: str = "vggish") -> Optional[float]:
    """Compute FAD between original and reconstructed audio directories."""
    try:
        from frechet_audio_distance import FrechetAudioDistance
    except ImportError:
        print("  WARNING: frechet-audio-distance not installed. Skipping FAD.")
        print("  Install with: pip install frechet-audio-distance")
        return None

    frechet = FrechetAudioDistance(
        model_name=model_name,
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
    )

    fad_score = frechet.score(orig_dir, recon_dir, dtype="float32")
    return float(fad_score)


# ---- Baseline evaluation ----

def evaluate_encodec_reconstruction(
    samples: List[Dict],
    device: torch.device,
    save_dir: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Evaluate EnCodec reconstruction SI-SDR."""
    try:
        from encodec import EncodecModel
    except ImportError:
        print("  encodec not installed. Run: pip install encodec")
        return None

    import torchaudio

    print("  Loading Encodec 48kHz model...")
    model = EncodecModel.encodec_model_48khz()
    model.to(device).eval()
    model.set_target_bandwidth(6.0)

    our_sr = 44100
    enc_sr = 48000

    si_sdr_list = []

    orig_dir = None
    recon_dir = None
    if save_dir:
        orig_dir = os.path.join(save_dir, "original")
        recon_dir = os.path.join(save_dir, "reconstructed_encodec")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

    for i, sample in enumerate(samples):
        x_wave = sample["x_wave"]
        if x_wave.dim() == 1:
            x_wave = x_wave.unsqueeze(0)  # [1, L]

        x_48 = torchaudio.transforms.Resample(our_sr, enc_sr)(x_wave).to(device)

        # EnCodec expects [B, C, L], 48kHz model is stereo
        x_48 = x_48.unsqueeze(0)  # [1, 1, L]
        if model.channels == 2:
            x_48 = x_48.expand(-1, 2, -1)

        with torch.no_grad():
            z = model.encoder(x_48)
            x_hat = model.decoder(z)

        # Convert back to mono and resample
        if x_hat.shape[1] == 2:
            x_hat = x_hat.mean(dim=1, keepdim=True)
        x_hat_44 = torchaudio.transforms.Resample(enc_sr, our_sr).to(device)(
            x_hat.squeeze(0)
        ).unsqueeze(0)

        # Original in 44.1kHz
        x_orig = x_wave.unsqueeze(0).to(device)  # [1, 1, L]

        min_len = min(x_hat_44.shape[-1], x_orig.shape[-1])
        sdr_val = si_sdr(
            x_hat_44[0, 0, :min_len].cpu(),
            x_orig[0, 0, :min_len].cpu(),
        )
        si_sdr_list.append(sdr_val)

        if save_dir:
            orig_audio = x_orig[0, 0, :min_len].cpu().numpy()
            recon_audio = x_hat_44[0, 0, :min_len].cpu().float().numpy()
            sf.write(
                os.path.join(orig_dir, f"sample_{i:04d}.wav"),
                orig_audio, SAMPLE_RATE, subtype="PCM_16",
            )
            sf.write(
                os.path.join(recon_dir, f"sample_{i:04d}.wav"),
                recon_audio, SAMPLE_RATE, subtype="PCM_16",
            )

    return {
        "SI-SDR_mean": float(np.mean(si_sdr_list)),
        "SI-SDR_std": float(np.std(si_sdr_list)),
        "SI-SDR_median": float(np.median(si_sdr_list)),
        "num_samples": len(samples),
    }


def evaluate_dac_reconstruction(
    samples: List[Dict],
    device: torch.device,
    save_dir: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Evaluate DAC reconstruction SI-SDR."""
    try:
        import dac
        from dac.utils import download
    except ImportError:
        print("  dac not installed. Run: pip install descript-audio-codec")
        return None

    print("  Loading DAC 44kHz model...")
    model_path = download(model_type="44khz")
    model = dac.DAC.load(model_path)
    model.to(device).eval()

    si_sdr_list = []

    orig_dir = None
    recon_dir = None
    if save_dir:
        orig_dir = os.path.join(save_dir, "original")
        recon_dir = os.path.join(save_dir, "reconstructed_dac")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

    for i, sample in enumerate(samples):
        x_wave = sample["x_wave"]
        if x_wave.dim() == 1:
            x_wave = x_wave.unsqueeze(0)

        x = x_wave.unsqueeze(0).to(device)  # [1, 1, L]

        with torch.no_grad():
            z, _, _, _, _ = model.encode(x)
            x_hat = model.decode(z)

        min_len = min(x_hat.shape[-1], x.shape[-1])
        sdr_val = si_sdr(
            x_hat[0, 0, :min_len].cpu(),
            x[0, 0, :min_len].cpu(),
        )
        si_sdr_list.append(sdr_val)

        if save_dir:
            orig_audio = x[0, 0, :min_len].cpu().numpy()
            recon_audio = x_hat[0, 0, :min_len].cpu().float().numpy()
            sf.write(
                os.path.join(orig_dir, f"sample_{i:04d}.wav"),
                orig_audio, SAMPLE_RATE, subtype="PCM_16",
            )
            sf.write(
                os.path.join(recon_dir, f"sample_{i:04d}.wav"),
                recon_audio, SAMPLE_RATE, subtype="PCM_16",
            )

    return {
        "SI-SDR_mean": float(np.mean(si_sdr_list)),
        "SI-SDR_std": float(np.std(si_sdr_list)),
        "SI-SDR_median": float(np.median(si_sdr_list)),
        "num_samples": len(samples),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Perceptual evaluation: FAD and SI-SDR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to main checkpoint (Phase 2)",
    )
    parser.add_argument(
        "--checkpoint2", type=str, default=None,
        help="Optional: Phase 1 checkpoint for comparison",
    )
    parser.add_argument(
        "--chunks-dir", type=str, default="./musdb-chunks-stft",
        help="Directory containing preprocessed MUSDB chunks",
    )
    parser.add_argument(
        "--num-samples", type=int, default=200,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--eval-baselines", action="store_true",
        help="Also evaluate EnCodec and DAC reconstruction",
    )
    parser.add_argument(
        "--output", type=str, default="results/perceptual_metrics.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--skip-fad", action="store_true",
        help="Skip FAD computation (only compute SI-SDR)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test samples
    print("Loading test samples...")
    samples = load_test_singles(args.chunks_dir, args.num_samples)
    print(f"Loaded {len(samples)} test samples\n")

    all_results = {}

    # Evaluate main checkpoint (Phase 2)
    print(f"Evaluating: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "phase2")
        metrics = evaluate_reconstruction(model, samples, device, save_dir=save_dir)

        if not args.skip_fad:
            print("  Computing FAD (VGGish)...")
            fad = compute_fad(
                os.path.join(save_dir, "original"),
                os.path.join(save_dir, "reconstructed"),
            )
            if fad is not None:
                metrics["FAD_vggish"] = fad
                print(f"  FAD (VGGish): {fad:.4f}")

        print(f"  SI-SDR: {metrics['SI-SDR_mean']:.2f} +/- {metrics['SI-SDR_std']:.2f} dB")
        all_results["phase2"] = metrics
        del model
        torch.cuda.empty_cache()

    # Evaluate Phase 1 if provided
    if args.checkpoint2:
        print(f"\nEvaluating: {args.checkpoint2}")
        model2 = load_model(args.checkpoint2, device)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "phase1")
            metrics2 = evaluate_reconstruction(model2, samples, device, save_dir=save_dir)

            if not args.skip_fad:
                print("  Computing FAD (VGGish)...")
                fad = compute_fad(
                    os.path.join(save_dir, "original"),
                    os.path.join(save_dir, "reconstructed"),
                )
                if fad is not None:
                    metrics2["FAD_vggish"] = fad
                    print(f"  FAD (VGGish): {fad:.4f}")

            print(f"  SI-SDR: {metrics2['SI-SDR_mean']:.2f} +/- {metrics2['SI-SDR_std']:.2f} dB")
            all_results["phase1"] = metrics2
            del model2
            torch.cuda.empty_cache()

    # Evaluate baselines
    if args.eval_baselines:
        with tempfile.TemporaryDirectory() as tmpdir:
            print("\nEvaluating EnCodec 48kHz...")
            enc_save_dir = os.path.join(tmpdir, "encodec")
            enc_results = evaluate_encodec_reconstruction(samples, device, save_dir=enc_save_dir)
            if enc_results:
                if not args.skip_fad:
                    print("  Computing FAD (VGGish)...")
                    fad = compute_fad(
                        os.path.join(enc_save_dir, "original"),
                        os.path.join(enc_save_dir, "reconstructed_encodec"),
                    )
                    if fad is not None:
                        enc_results["FAD_vggish"] = fad
                        print(f"  FAD (VGGish): {fad:.4f}")
                print(f"  SI-SDR: {enc_results['SI-SDR_mean']:.2f} +/- {enc_results['SI-SDR_std']:.2f} dB")
                all_results["encodec_48khz"] = enc_results
            torch.cuda.empty_cache()

            print("\nEvaluating DAC 44kHz...")
            dac_save_dir = os.path.join(tmpdir, "dac")
            dac_results = evaluate_dac_reconstruction(samples, device, save_dir=dac_save_dir)
            if dac_results:
                if not args.skip_fad:
                    print("  Computing FAD (VGGish)...")
                    fad = compute_fad(
                        os.path.join(dac_save_dir, "original"),
                        os.path.join(dac_save_dir, "reconstructed_dac"),
                    )
                    if fad is not None:
                        dac_results["FAD_vggish"] = fad
                        print(f"  FAD (VGGish): {fad:.4f}")
                print(f"  SI-SDR: {dac_results['SI-SDR_mean']:.2f} +/- {dac_results['SI-SDR_std']:.2f} dB")
                all_results["dac_44khz"] = dac_results

    # Print summary
    print("\n" + "=" * 70)
    print("PERCEPTUAL METRICS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'SI-SDR (dB)':<18} {'FAD (VGGish)':<15}")
    print("-" * 70)
    for name, metrics in all_results.items():
        sdr = f"{metrics['SI-SDR_mean']:.2f} +/- {metrics['SI-SDR_std']:.2f}"
        fad = f"{metrics.get('FAD_vggish', 'N/A'):.4f}" if "FAD_vggish" in metrics else "N/A"
        print(f"{name:<25} {sdr:<18} {fad:<15}")
    print("=" * 70)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
