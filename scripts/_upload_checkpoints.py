"""One-off: back up inference checkpoints to HuggingFace Hub (account SoMa25).

Uploads every musicgen best.pth and every M2L *_ema.pt with clean repo paths,
plus a MANIFEST.md mapping each file to its run. Private repo by default.
"""

import glob
import os
from pathlib import Path

import argparse

from huggingface_hub import HfApi

REPO_ID = os.environ.get("MUSICGEN_HF_REPO", "SoMa25/mixing-equivariant-ae-checkpoints")
MUSICGEN = os.environ.get("MUSICGEN_REPO",
                          str(Path(__file__).resolve().parents[1]))
M2L = os.environ.get("MUSICGEN_M2L_REPO",
                     str(Path(__file__).resolve().parents[1].parent / "music2latent-mix"))

# The paper links these weights, so the repo must be public. Set
# MUSICGEN_HF_PRIVATE=1 to keep a working copy private instead.
PRIVATE = os.environ.get("MUSICGEN_HF_PRIVATE", "") == "1"

def plan():
    """Build the (local, remote, note) list. Pure: touches nothing remote."""
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
    # Several timestamped runs of one phase would collide on the same remote
    # name, silently publishing whichever went last. Keep the newest and say so.
    m2l = {}
    for d in sorted(glob.glob(f"{M2L}/checkpoints/*/*/*_ema.pt")):
        phase = os.path.basename(os.path.dirname(os.path.dirname(d)))
        m2l.setdefault(phase, []).append(d)
    for phase, paths in sorted(m2l.items()):
        chosen = max(paths, key=os.path.getmtime)
        if len(paths) > 1:
            print(f"[collision] {phase}: {len(paths)} candidates map to "
                  f"m2l/{phase}_ema.pt; taking the newest:")
            for q in paths:
                print(f"    {'*' if q == chosen else ' '} {q}")
        uploads.append((chosen, f"m2l/{phase}_ema.pt", f"M2L {phase} (EMA-merged)"))
    return uploads

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--private", action="store_true",
                    help="create the repo private (default: public, because "
                         "the paper links these weights)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the upload plan and exit, touching nothing")
    a = ap.parse_args()

    uploads = plan()
    total = sum(os.path.getsize(l) for l, _, _ in uploads) / 1e9
    print(f"plan: {len(uploads)} files, {total:.1f} GB -> {a.repo_id} "
          f"({'private' if a.private else 'public'})")
    for local, repo_path, note in uploads:
        print(f"  -> {repo_path}  ({os.path.getsize(local)/1e9:.2f} GB)  [{note}]")
    if a.dry_run:
        print("\n[dry-run] nothing uploaded")
        return
    if not uploads:
        raise SystemExit("nothing to upload")

    api = HfApi()
    api.create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)
    for local, repo_path, note in uploads:
        print(f"  uploading {repo_path}", flush=True)
        api.upload_file(path_or_fileobj=local, path_in_repo=repo_path,
                        repo_id=a.repo_id, repo_type="model")

    manifest = "# Checkpoint manifest\n\nInference weights for the mixing-equivariant AE paper.\n\n"
    for local, repo_path, note in uploads:
        manifest += f"- `{repo_path}` — {note}\n"
    manifest += ("\nmusicgen files load via `evaluation/compute_*.py --checkpoint`; "
                 "M2L files via `evaluation/m2l_run_*.py --m2l-checkpoint`.\n")
    api.upload_file(path_or_fileobj=manifest.encode(), path_in_repo="MANIFEST.md",
                    repo_id=a.repo_id, repo_type="model")
    print(f"\nDONE: https://huggingface.co/{a.repo_id}")


# Importing this module must not publish anything.
if __name__ == "__main__":
    main()
