"""Same-latent repeated-decoding control for the decode-noise protocol (Fig. 1).

Fig. 1 shows the decode-vs-decode SI-SDR of a consistency autoencoder swinging
by ~12 dB between shared and independent decoder noise. That demonstrates
protocol sensitivity, not by itself that phase (decoder noise) is the cause.
This control removes latent arithmetic entirely: decode the SAME latent twice
with independent noise and score the two decodes against each other. If that
alone lands near the independent-noise Fig. 1 number, the disagreement is a
property of the stochastic decoder, independent of any latent operation.

Reports, on the same MUSDB chunks:
  same_latent_indep   SI-SDR(g_1(z), g_2(z)), independent noise, same z
  same_latent_shared  SI-SDR(g(z), g(z)), shared noise (sanity: identical)
  lin_shared / lin_indep   the Fig. 1 quantities on these chunks

Usage:
  python -m scripts._diag_same_latent --checkpoint checkpoints/m2l/music2latent.pt
"""
import argparse, json, statistics, sys
import torch
sys.path.insert(0, '.')
from data.dataset import WaveformDataset
from torch.utils.data import DataLoader, Subset
from training.config import get_device
from evaluation.compute_mixing_metrics import _process_batch, _si_sdr
from evaluation.m2l_adapter import M2LAutoencoderAdapter, _M2LDecoderWrapper
from scripts._diag_old_vs_new_eval import patch_adapter_random_noise

BATCH = 8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--max-batches', type=int, default=30)
    ap.add_argument('--out', default='evaluation/v2_metrics/same_latent_control.json')
    a = ap.parse_args()
    dev = get_device()
    ds = WaveformDataset('./chunks-44k-1s', split='test')
    idx = [i for f in ds.files if f.get('source') == 'musdb' for i in range(f['start'], f['end'])]
    sub = Subset(ds, idx[:a.max_batches * BATCH])
    dl = DataLoader(sub, batch_size=BATCH, shuffle=False, num_workers=0,
                    collate_fn=lambda b: {'x_wave': torch.stack([x['x_wave'] for x in b]),
                                          'source': [x['source'] for x in b]})
    torch.manual_seed(0)
    ad = M2LAutoencoderAdapter(m2l_checkpoint_path=a.checkpoint, device=dev).to(dev).eval()
    L = ad.decoder.target_length
    dec1 = _M2LDecoderWrapper(ad.gen, L, noise_seed=1).to(dev)
    dec2 = _M2LDecoderWrapper(ad.gen, L, noise_seed=2).to(dev)
    same_ind, same_sh, lin_sh, lin_ind = [], [], [], []
    with torch.no_grad():
        for bi, b in enumerate(dl):
            if bi >= a.max_batches: break
            x = b['x_wave'].to(dev)
            if x.size(0) < 2: continue
            z = ad.encoder(x)
            g1, _ = dec1(z); g2, _ = dec2(z); g1b, _ = dec1(z)
            T = min(g1.size(-1), g2.size(-1))
            same_ind += _si_sdr(g1[..., :T], g2[..., :T]).cpu().tolist()
            same_sh += _si_sdr(g1[..., :T], g1b[..., :T]).cpu().tolist()
            lin_sh += [v for _, v in _process_batch(ad, x, list(b['source']), alpha=0.5)['sdr_lin']]
    adr = patch_adapter_random_noise(ad)
    with torch.no_grad():
        for bi, b in enumerate(dl):
            if bi >= a.max_batches: break
            x = b['x_wave'].to(dev)
            if x.size(0) < 2: continue
            lin_ind += [v for _, v in _process_batch(adr, x, list(b['source']), alpha=0.5)['sdr_lin']]
    res = {k: {'mean': statistics.mean(v), 'std': statistics.stdev(v), 'n': len(v)}
           for k, v in [('same_latent_indep', same_ind), ('same_latent_shared', same_sh),
                        ('lin_shared', lin_sh), ('lin_indep', lin_ind)]}
    for k, r in res.items():
        print(f"{k:20s} {r['mean']:+8.2f} dB  (sd {r['std']:.2f}, n={r['n']})")
    json.dump({'checkpoint': a.checkpoint, 'results': res}, open(a.out, 'w'), indent=2)
    print('wrote', a.out)

if __name__ == '__main__':
    main()
