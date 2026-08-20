"""Diagnose the SI-SDR_lin discrepancy between the paper's M2L baseline (-8.7)
and our new measurement (+5.02).

Hypothesis: the old eval used INDEPENDENT random noise per decode call. The
two decoded waveforms (g(z̄) and g(f(x̄))) had independent random phases,
so SI-SDR was dominated by phase variance, not real linearity. The U-shape
in α observed in results/baseline_evaluation.json is the smoking gun.

Test: run both Phase 0 (vanilla M2L) and Phase 2 (our fine-tune) checkpoints
under TWO eval protocols on the same small subset of MUSDB chunks:
  (A) shared-noise protocol  (current adapter, fixed seed within batch)
  (B) per-call independent noise  (matches old paper's protocol)

If (B) gives ~-8.7 for vanilla M2L AND a much higher number for Phase 2 → the
old "-8.7 → 11.7" jump is mostly phase-noise reduction, not real linearity.
If (A) shows a consistent improvement of ~Phase 2's +1.89 dB across protocols
→ the gain is real but smaller than the paper claims.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, '.')

from data.dataset import WaveformDataset
from training.config import get_device
from evaluation.compute_mixing_metrics import _process_batch, _no_fixed_point_perm, _si_sdr
from evaluation.m2l_adapter import M2LAutoencoderAdapter, _M2LDecoderWrapper

# Checkpoints live outside this repo. Override either with an env var:
#   export MUSICGEN_M2L_PUBLISHED=/path/to/music2latent.pt        # Pasini et al.
#   export MUSICGEN_M2L_CHECKPOINT=/path/to/model_..._50000.pt    # our Phase 2
_M2L_REPO = Path(os.environ.get("MUSICGEN_M2L_REPO", Path(__file__).resolve().parents[1].parent / "music2latent-mix"))
PHASE0_CKPT = os.environ.get(
    "MUSICGEN_M2L_PUBLISHED",
    str(Path(__file__).resolve().parents[1].parent / "music2latent" / "music2latent" / "models" / "music2latent.pt"))
PHASE2_CKPT = os.environ.get(
    "MUSICGEN_M2L_CHECKPOINT",
    str(_M2L_REPO / "checkpoints" / "mix_phase2_decmix_consmix"
        / "2026-05-06_14-11-27" / "model_fid_-1.0_loss_119.60_iters_50000.pt"))
N_BATCHES = 25
BATCH_SIZE = 8


def patch_adapter_random_noise(adapter):
    """Monkey-patch the decoder wrapper to use independent random noise each call,
    mimicking the old paper's protocol."""
    old_decoder = adapter.decoder
    gen = adapter.gen
    target_length = old_decoder.target_length

    from music2latent import hparams as hp
    from music2latent.audio import to_waveform

    class RandomNoiseDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gen = gen
            self.target_length = target_length

        def forward(self, z):
            B = z.size(0)
            downscaling = 2 ** hp.freq_downsample_list.count(0)
            T_stft = int(z.shape[-1] * downscaling)
            initial_noise = torch.randn(
                (B, hp.data_channels, hp.hop * 2, T_stft),
                device=z.device, dtype=torch.float32,
            ) * hp.sigma_max
            if z.dtype != torch.float32:
                initial_noise = initial_noise.to(z.dtype)
            pyramid = self.gen.decoder(z)
            x_stft = self.gen(z, initial_noise, sigma=hp.sigma_max, pyramid_latents=pyramid)
            x_wave = to_waveform(x_stft)
            x_wave = x_wave[:, :self.target_length]
            return x_wave.unsqueeze(1), None

    adapter.decoder = RandomNoiseDecoder()
    return adapter


def run_eval(checkpoint_path, protocol, val_loader, device, max_batches, source_filter=None):
    print(f"\n=== checkpoint={checkpoint_path}, protocol={protocol} ===")
    torch.manual_seed(0)
    adapter = M2LAutoencoderAdapter(m2l_checkpoint_path=checkpoint_path, device=device).to(device)
    adapter.eval()
    if protocol == "random":
        adapter = patch_adapter_random_noise(adapter)

    sdr_rec_all, sdr_lin_all = [], []
    for bi, batch in enumerate(val_loader):
        if bi >= max_batches: break
        x_wave = batch["x_wave"].to(device, non_blocking=True)
        sources = batch.get("source", ["?"] * x_wave.size(0))
        if source_filter is not None:
            keep = [i for i, s in enumerate(sources) if s == source_filter]
            if not keep: continue
            x_wave = x_wave[keep]
            sources = [sources[i] for i in keep]
        if x_wave.size(0) < 2: continue
        out = _process_batch(adapter, x_wave, list(sources), alpha=0.5)
        for src, val in out['sdr_rec']: sdr_rec_all.append(val)
        for src, val in out['sdr_lin']: sdr_lin_all.append(val)

    n = len(sdr_lin_all)
    if n == 0:
        print(f"  no samples evaluated"); return None, None
    import statistics
    mean_lin = statistics.mean(sdr_lin_all); std_lin = statistics.stdev(sdr_lin_all) if n > 1 else 0.0
    mean_rec = statistics.mean(sdr_rec_all); std_rec = statistics.stdev(sdr_rec_all) if n > 1 else 0.0
    print(f"  n={n}, sdr_rec={mean_rec:+.3f} ± {std_rec:.3f}, sdr_lin={mean_lin:+.3f} ± {std_lin:.3f}")
    return mean_rec, mean_lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='musdb', help='Filter to this source (default: musdb)')
    ap.add_argument('--max-batches', type=int, default=N_BATCHES)
    args = ap.parse_args()
    device = get_device()

    # Build a MUSDB-only loader directly from the index — match the old paper's
    # eval which used 200 MUSDB stem pairs.
    from data.dataset import WaveformDataset
    from torch.utils.data import DataLoader, Subset
    val_ds = WaveformDataset('./chunks-44k-1s', split='test')
    musdb_idxs = []
    for f in val_ds.files:
        if f.get('source') == args.source:
            musdb_idxs.extend(range(f['start'], f['end']))
    print(f"Found {len(musdb_idxs)} {args.source} chunks in test split")
    sub = Subset(val_ds, musdb_idxs[:args.max_batches * BATCH_SIZE])
    val_loader = DataLoader(sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                            pin_memory=(device.type=='cuda'),
                            collate_fn=lambda batch: {
                                'x_wave': torch.stack([b['x_wave'] for b in batch]),
                                'source': [b['source'] for b in batch],
                            })

    print(f"Filtering to source: {args.source}")
    print(f"Eval batches: {args.max_batches} × batch={BATCH_SIZE}\n")

    # source_filter=None now since we already filtered the loader
    rec_p0_a, lin_p0_a = run_eval(PHASE0_CKPT, "shared",  val_loader, device, args.max_batches, source_filter=None)
    rec_p2_a, lin_p2_a = run_eval(PHASE2_CKPT, "shared",  val_loader, device, args.max_batches, source_filter=None)
    rec_p0_b, lin_p0_b = run_eval(PHASE0_CKPT, "random",  val_loader, device, args.max_batches, source_filter=None)
    rec_p2_b, lin_p2_b = run_eval(PHASE2_CKPT, "random",  val_loader, device, args.max_batches, source_filter=None)

    print("\n========== summary ==========")
    print(f"{'protocol':12s}  {'phase':6s}  {'sdr_rec':>10s}  {'sdr_lin':>10s}")
    print("-" * 50)
    print(f"{'shared':12s}  {'P0':6s}  {rec_p0_a:+10.3f}  {lin_p0_a:+10.3f}")
    print(f"{'shared':12s}  {'P2':6s}  {rec_p2_a:+10.3f}  {lin_p2_a:+10.3f}")
    print(f"{'random':12s}  {'P0':6s}  {rec_p0_b:+10.3f}  {lin_p0_b:+10.3f}")
    print(f"{'random':12s}  {'P2':6s}  {rec_p2_b:+10.3f}  {lin_p2_b:+10.3f}")
    print()
    if lin_p0_b is not None and lin_p2_b is not None:
        print(f"random-noise P0 -> P2: {lin_p0_b:+.2f} -> {lin_p2_b:+.2f}  (Δ={lin_p2_b-lin_p0_b:+.2f})")
    if lin_p0_a is not None and lin_p2_a is not None:
        print(f"shared-noise P0 -> P2: {lin_p0_a:+.2f} -> {lin_p2_a:+.2f}  (Δ={lin_p2_a-lin_p0_a:+.2f})")


if __name__ == '__main__':
    main()
