"""Verify state_dict load actually populated the model (not silently skipped)."""
import os
from pathlib import Path

import torch
from music2latent.inference import EncoderDecoder
from music2latent.models import UNet
from training.config import get_device


def main():
    device = get_device()

    # Load fresh UNet, see baseline weight stats
    fresh = UNet().to(device)
    fresh_w_mean = fresh.encoder.conv_inp.weight.float().mean().item()
    fresh_w_std = fresh.encoder.conv_inp.weight.float().std().item()
    print(f"FRESH encoder.conv_inp: mean={fresh_w_mean:+.4e}  std={fresh_w_std:.4e}")

    # Load the checkpoint via EncoderDecoder
    ed = EncoderDecoder(device=device)
    loaded_w_mean = ed.gen.encoder.conv_inp.weight.float().mean().item()
    loaded_w_std = ed.gen.encoder.conv_inp.weight.float().std().item()
    print(f"LOADED encoder.conv_inp: mean={loaded_w_mean:+.4e}  std={loaded_w_std:.4e}")

    # Load the raw state_dict and compare key sets
    # Override with: export MUSICGEN_M2L_PUBLISHED=/path/to/music2latent.pt
    ckpt_path = os.environ.get(
        "MUSICGEN_M2L_PUBLISHED",
        str(Path(__file__).resolve().parents[1].parent
            / "music2latent" / "music2latent" / "models" / "music2latent.pt"))
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("checkpoint keys:", list(sd.keys()))
    if "gen_state_dict" in sd:
        gsd = sd["gen_state_dict"]
        print("gen_state_dict has", len(gsd), "tensors")
        ckpt_keys = set(gsd.keys())
        model_keys = set(ed.gen.state_dict().keys())
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        print(f"  missing in ckpt:    {len(missing)}  e.g. {list(missing)[:3]}")
        print(f"  unexpected in ckpt: {len(unexpected)}  e.g. {list(unexpected)[:3]}")
    if "ema_state_dict" in sd:
        print("HAS ema_state_dict — published checkpoint includes it; the master inference path"
              " does NOT use it (only loads gen_state_dict)")
        ema = sd["ema_state_dict"]
        print("  ema keys:", list(ema.keys())[:5])


if __name__ == "__main__":
    main()
