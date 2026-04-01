"""
Evaluate external baselines (Music2Latent, EnCodec, DAC) with MixRate alpha sweep.

Usage:
    python -m evaluation.baseline_eval --model music2latent
    python -m evaluation.baseline_eval --model encodec
    python -m evaluation.baseline_eval --model dac
    python -m evaluation.baseline_eval --model all
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.utils import load_model, load_test_stem_pairs

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Reference model checkpoint — used only for its MRSTFT loss function
REF_CHECKPOINT = "checkpoints/comp-24x-v6/best_model.pth"


def _get_loss_fn(device):
    """Load our model to use its mrstft_loss for fair comparison."""
    ref_model = load_model(REF_CHECKPOINT, device)
    return ref_model


def compute_total_loss(x_pred, x_target, ref_model):
    """Compute L1 + MRSTFT loss (same as our model's MixRate evaluation)."""
    # Ensure 3D: [B, 1, L]
    if x_pred.dim() == 1:
        x_pred = x_pred.unsqueeze(0).unsqueeze(0)
    elif x_pred.dim() == 2:
        x_pred = x_pred.unsqueeze(0)
    if x_target.dim() == 1:
        x_target = x_target.unsqueeze(0).unsqueeze(0)
    elif x_target.dim() == 2:
        x_target = x_target.unsqueeze(0)
    with torch.no_grad():
        l1 = F.l1_loss(x_pred, x_target).item()
        mr = ref_model.mrstft_loss(x_pred, x_target).item()
    return l1 + mr


def _to_tensor_1d(x, device):
    """Convert Music2Latent output (tensor or numpy, 1D or 2D) to 1D torch tensor."""
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if x.ndim == 2:
        # Music2Latent returns [channels, samples] — take first channel
        x = x[0]
    return torch.from_numpy(x.copy()).float().to(device)


def evaluate_music2latent(pairs, alphas, device):
    """Evaluate Music2Latent baseline with MRSTFT+L1 loss."""
    from music2latent import EncoderDecoder

    m2l = EncoderDecoder(device=device)
    ref_model = _get_loss_fn(device)

    results = {}
    for alpha in alphas:
        beta = 1.0 - alpha
        rates = []
        interp_losses = []
        oracle_losses = []

        for i, sample in enumerate(pairs):
            x1_wave = sample["x1_wave"].numpy().squeeze()  # [L]
            x2_wave = sample["x2_wave"].numpy().squeeze()

            z1 = m2l.encode(x1_wave)
            z2 = m2l.encode(x2_wave)

            x_mix = alpha * x1_wave + beta * x2_wave

            z_interp = alpha * z1 + beta * z2
            x_interp = m2l.decode(z_interp)

            z_real = m2l.encode(x_mix)
            x_oracle = m2l.decode(z_real)

            # Convert to 1D tensors
            x_mix_t = torch.from_numpy(x_mix.copy()).float().to(device)
            x_interp_t = _to_tensor_1d(x_interp, device)
            x_oracle_t = _to_tensor_1d(x_oracle, device)

            # Trim to same length
            min_len = min(len(x_mix_t), len(x_interp_t), len(x_oracle_t))
            x_mix_t = x_mix_t[:min_len]
            x_interp_t = x_interp_t[:min_len]
            x_oracle_t = x_oracle_t[:min_len]

            loss_interp = compute_total_loss(x_interp_t, x_mix_t, ref_model)
            loss_oracle = compute_total_loss(x_oracle_t, x_mix_t, ref_model)

            rate = loss_interp / (loss_oracle + 1e-8)
            rates.append(rate)
            interp_losses.append(loss_interp)
            oracle_losses.append(loss_oracle)

            if (i + 1) % 50 == 0:
                print(f"    alpha={alpha}, {i+1}/{len(pairs)}, "
                      f"MixRate={np.mean(rates):.4f}, "
                      f"interp={np.mean(interp_losses):.4f}, "
                      f"oracle={np.mean(oracle_losses):.4f}")

        results[alpha] = {
            "MixRate_mean": float(np.mean(rates)),
            "MixRate_std": float(np.std(rates)),
            "MixRate_median": float(np.median(rates)),
            "MixRate_p90": float(np.percentile(rates, 90)),
            "MixReconInterp_mean": float(np.mean(interp_losses)),
            "MixReconReal_mean": float(np.mean(oracle_losses)),
            "num_samples": len(pairs),
        }
        print(f"  alpha={alpha}: MixRate={results[alpha]['MixRate_mean']:.4f} "
              f"(std={results[alpha]['MixRate_std']:.4f})")

    return results


def evaluate_encodec(pairs, alphas, device):
    """Evaluate EnCodec-48kHz baseline with MRSTFT+L1 loss."""
    from encodec import EncodecModel

    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(24.0)
    model.to(device).eval()
    ref_model = _get_loss_fn(device)

    results = {}
    for alpha in alphas:
        beta = 1.0 - alpha
        rates = []

        for i, sample in enumerate(pairs):
            x1 = sample["x1_wave"].unsqueeze(0).to(device)
            x2 = sample["x2_wave"].unsqueeze(0).to(device)

            if x1.dim() == 2:
                x1 = x1.unsqueeze(0)
                x2 = x2.unsqueeze(0)

            if x1.shape[1] == 1:
                x1 = x1.repeat(1, 2, 1)
                x2 = x2.repeat(1, 2, 1)

            # Resample 44100 -> 48000
            x1_48k = F.interpolate(x1, scale_factor=48000/44100, mode='linear')
            x2_48k = F.interpolate(x2, scale_factor=48000/44100, mode='linear')
            x_mix_48k = alpha * x1_48k + beta * x2_48k

            with torch.no_grad():
                z1 = model.encoder(x1_48k)
                z2 = model.encoder(x2_48k)
                z_mix = model.encoder(x_mix_48k)

                z_interp = alpha * z1 + beta * z2
                x_interp = model.decoder(z_interp)
                x_oracle = model.decoder(z_mix)

            # Use first channel, compute MRSTFT+L1
            min_len = min(x_mix_48k.shape[-1], x_interp.shape[-1], x_oracle.shape[-1])
            mix_mono = x_mix_48k[:, 0:1, :min_len]
            interp_mono = x_interp[:, 0:1, :min_len]
            oracle_mono = x_oracle[:, 0:1, :min_len]

            loss_interp = compute_total_loss(interp_mono, mix_mono, ref_model)
            loss_oracle = compute_total_loss(oracle_mono, mix_mono, ref_model)

            rate = loss_interp / (loss_oracle + 1e-8)
            rates.append(rate)

        results[alpha] = {
            "MixRate_mean": float(np.mean(rates)),
            "MixRate_std": float(np.std(rates)),
            "MixRate_median": float(np.median(rates)),
            "MixRate_p90": float(np.percentile(rates, 90)),
            "num_samples": len(pairs),
        }
        print(f"  alpha={alpha}: MixRate={results[alpha]['MixRate_mean']:.4f}")

    return results


def evaluate_dac(pairs, alphas, device):
    """Evaluate DAC-44kHz baseline with MRSTFT+L1 loss."""
    import dac

    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path).to(device).eval()
    ref_model = _get_loss_fn(device)

    results = {}
    for alpha in alphas:
        beta = 1.0 - alpha
        rates = []

        for i, sample in enumerate(pairs):
            x1 = sample["x1_wave"].unsqueeze(0).to(device)
            x2 = sample["x2_wave"].unsqueeze(0).to(device)

            if x1.dim() == 2:
                x1 = x1.unsqueeze(0)
                x2 = x2.unsqueeze(0)

            x_mix = alpha * x1 + beta * x2

            with torch.no_grad():
                z1, _, _, _, _ = model.encode(x1)
                z2, _, _, _, _ = model.encode(x2)
                z_mix_real, _, _, _, _ = model.encode(x_mix)

                z_interp = alpha * z1 + beta * z2
                x_interp = model.decode(z_interp)
                x_oracle = model.decode(z_mix_real)

            min_len = min(x_mix.shape[-1], x_interp.shape[-1], x_oracle.shape[-1])
            loss_interp = compute_total_loss(
                x_interp[..., :min_len], x_mix[..., :min_len], ref_model
            )
            loss_oracle = compute_total_loss(
                x_oracle[..., :min_len], x_mix[..., :min_len], ref_model
            )

            rate = loss_interp / (loss_oracle + 1e-8)
            rates.append(rate)

        results[alpha] = {
            "MixRate_mean": float(np.mean(rates)),
            "MixRate_std": float(np.std(rates)),
            "MixRate_median": float(np.median(rates)),
            "MixRate_p90": float(np.percentile(rates, 90)),
            "num_samples": len(pairs),
        }
        print(f"  alpha={alpha}: MixRate={results[alpha]['MixRate_mean']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate external baselines with MixRate"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["music2latent", "encodec", "dac", "all"],
    )
    parser.add_argument(
        "--chunks-dir", type=str, default="./musdb-chunks-stft-5s",
    )
    parser.add_argument(
        "--num-samples", type=int, default=200,
    )
    parser.add_argument(
        "--output", type=str, default="results/baseline_evaluation.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test data
    print(f"Loading test stem pairs from {args.chunks_dir}...")
    pairs = load_test_stem_pairs(args.chunks_dir, args.num_samples)
    print(f"Loaded {len(pairs)} test stem pairs")

    # Load existing results if any
    all_results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {args.output}")

    models_to_eval = (
        ["music2latent", "encodec", "dac"]
        if args.model == "all"
        else [args.model]
    )

    for model_name in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        try:
            if model_name == "music2latent":
                results = evaluate_music2latent(pairs, ALPHAS, device)
            elif model_name == "encodec":
                results = evaluate_encodec(pairs, ALPHAS, device)
            elif model_name == "dac":
                results = evaluate_dac(pairs, ALPHAS, device)

            # Convert keys to strings for JSON
            all_results[model_name] = {
                str(k): v for k, v in results.items()
            }

            # Save incrementally
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved to {args.output}")

        except Exception as e:
            print(f"  ERROR evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY: MixRate @ alpha=0.5")
    print(f"{'='*80}")
    for name, res in all_results.items():
        rate = res.get("0.5", {}).get("MixRate_mean", float("nan"))
        print(f"  {name}: {rate:.4f}")


if __name__ == "__main__":
    main()
