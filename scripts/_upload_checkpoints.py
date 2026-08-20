"""One-off: back up inference checkpoints to HuggingFace Hub (account SoMa25).

Uploads every musicgen best.pth and every M2L *_ema.pt with clean repo paths,
plus a MANIFEST.md mapping each file to its run. Private repo by default.
"""

import glob
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = os.environ.get("MUSICGEN_HF_REPO", "SoMa25/mixing-equivariant-ae-checkpoints")
MUSICGEN = os.environ.get("MUSICGEN_REPO",
                          str(Path(__file__).resolve().parents[1]))
M2L = os.environ.get("MUSICGEN_M2L_REPO",
                     str(Path(__file__).resolve().parents[1].parent / "music2latent-mix"))

# The paper links these weights, so the repo must be public. Set
# MUSICGEN_HF_PRIVATE=1 to keep a working copy private instead.
PRIVATE = os.environ.get("MUSICGEN_HF_PRIVATE", "") == "1"

api = HfApi()
api.create_repo(REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)

uploads = []  # (local_path, repo_path, note)

# musicgen best.pth per run
for d in sorted(glob.glob(f"{MUSICGEN}/checkpoints/*/best.pth")):
    run = os.path.basename(os.path.dirname(d))
    if run == "clap":   # third-party LAION-CLAP, reproducible — skip
        continue
    uploads.append((d, f"musicgen/{run}/best.pth", f"musicgen run {run}"))

# M2L EMA-merged checkpoints (what the evals use).
# Path: .../checkpoints/<phase>/<timestamp>/<file>_ema.pt — take the phase dir
# two levels up (separator-agnostic; glob returns backslashes on Windows).
for d in sorted(glob.glob(f"{M2L}/checkpoints/*/*/*_ema.pt")):
    phase = os.path.basename(os.path.dirname(os.path.dirname(d)))
    uploads.append((d, f"m2l/{phase}_ema.pt", f"M2L {phase} (EMA-merged)"))

print(f"uploading {len(uploads)} files to {REPO_ID}")
for local, repo_path, note in uploads:
    sz = os.path.getsize(local) / 1e9
    print(f"  -> {repo_path}  ({sz:.2f} GB)  [{note}]", flush=True)
    api.upload_file(path_or_fileobj=local, path_in_repo=repo_path,
                    repo_id=REPO_ID, repo_type="model")

# manifest
manifest = "# Checkpoint manifest\n\nInference weights for the mixing-equivariant AE paper.\n\n"
for _, repo_path, note in uploads:
    manifest += f"- `{repo_path}` — {note}\n"
manifest += ("\nmusicgen files load via `evaluation/compute_*.py --checkpoint`; "
             "M2L files via `evaluation/m2l_run_*.py --m2l-checkpoint`.\n")
api.upload_file(path_or_fileobj=manifest.encode(), path_in_repo="MANIFEST.md",
                repo_id=REPO_ID, repo_type="model")
print(f"\nDONE: https://huggingface.co/{REPO_ID}")
