"""Adapters that expose third-party audio autoencoders behind the small
``model.encoder(wave) -> z`` / ``model.decoder(z) -> (wave, None)`` surface
that ``evaluation.compute_subtraction.run_eval`` needs.

None of these models were trained by us and none of them (except the Lin-CAE
family) was trained with any linearity objective. They exist to test whether
recentering the latent origin at the encoding of silence, f(0), changes
latent-subtraction scores on models we had no hand in.

Latent route. For the RVQ codecs (DAC, EnCodec) arithmetic is done on the
continuous encoder output *before* quantization and the decoder is fed that
continuous tensor directly; the quantizer is bypassed on every path, including
the reconstruction reference g(f(x)), so the comparison is self-consistent.

Sample rate. The eval works on 1 s mono chunks at 44.1 kHz. EnCodec-24k is
resampled 44.1 -> 24 kHz on the way in and 24 -> 44.1 kHz on the way out; its
outputs are therefore band-limited to 12 kHz, which lowers its reconstruction
reference and every route equally.

Decoder noise. The Lin-CAE family are consistency autoencoders whose decoder
starts from Gaussian noise drawn from the global torch RNG. ``run_eval``
reseeds that RNG before each decode, so the two decodes being compared share
their noise; see ``m2l_adapter.py`` for why that matters.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

SR = 44100
CHUNK_LEN = SR

MODEL_IDS = {
    "dac44k": "descript/dac_44khz",
    "encodec24k": "facebook/encodec_24khz",
    "lincae": "lin-cae",
    "lincae2": "lin-cae-2",
    "lincae-m2l": "m2l",
}


class _Enc(nn.Module):
    def __init__(self, fn):
        super().__init__(); self.fn = fn
    def forward(self, x):  # x: [B, 1, T] or [B, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.fn(x.float())


class _Dec(nn.Module):
    def __init__(self, fn):
        super().__init__(); self.fn = fn
    def forward(self, z) -> Tuple[torch.Tensor, None]:
        y = self.fn(z)
        if y.dim() == 2:
            y = y.unsqueeze(1)
        return y[..., :CHUNK_LEN], None


class CodecAdapter(nn.Module):
    """Holds the third-party model and the two callables the eval uses."""

    def __init__(self, name: str, device: torch.device):
        super().__init__()
        if name not in MODEL_IDS:
            raise ValueError(f"unknown model {name}; choose from {sorted(MODEL_IDS)}")
        self.name = name
        self.hf_id = MODEL_IDS[name]
        self.latent_route = "continuous"
        if name == "dac44k":
            from transformers.models.dac.modeling_dac import DacModel
            m = DacModel.from_pretrained(self.hf_id).to(device).eval()
            self.model = m
            self.latent_route = "continuous encoder output, quantizer bypassed"
            self.encoder = _Enc(lambda x: m.encoder(x))
            self.decoder = _Dec(lambda z: m.decoder(z))
        elif name == "encodec24k":
            import torchaudio
            from transformers.models.encodec.modeling_encodec import EncodecModel
            m = EncodecModel.from_pretrained(self.hf_id).to(device).eval()
            self.model = m
            self.latent_route = "continuous encoder output, quantizer bypassed; 44.1<->24 kHz resampling"
            down = torchaudio.transforms.Resample(SR, 24000).to(device)
            up = torchaudio.transforms.Resample(24000, SR).to(device)
            self.encoder = _Enc(lambda x: m.encoder(down(x)))
            self.decoder = _Dec(lambda z: up(m.decoder(z)))
        else:
            from linear_cae import Autoencoder
            m = Autoencoder.from_pretrained(self.hf_id, max_batch_size=256).to(device).eval()
            self.model = m
            self.latent_route = "Lin-CAE latent (scale_factor applied inside encode/decode)"
            self.encoder = _Enc(lambda x: m.encode(x.squeeze(1)))
            self.decoder = _Dec(lambda z: m.decode(z, full_length=CHUNK_LEN))

    def describe(self) -> dict:
        n = sum(p.numel() for p in self.model.parameters())
        return {"model": self.name, "hf_id": self.hf_id, "params": int(n),
                "latent_route": self.latent_route}
