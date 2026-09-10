"""Shared run resolution for the batch evaluation drivers.

Two release problems this fixes.

**Config resolution.** The published weights are distributed as
`checkpoints/<run>/best.pth` alone, but the drivers looked for
`checkpoints/<run>/config.yaml`, which only exists on a machine that did the
training. A fresh clone therefore skipped every model and still exited 0. Each
run now falls back to its tracked experiment config, so the documented
download workflow actually evaluates something.

**Silent skipping.** Missing inputs were counted as skips, never failures. A
run named explicitly with `--only` is now a hard error if it cannot be
evaluated, and the drivers exit non-zero if anything failed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# run name -> tracked experiment config. The training config is what defines
# the architecture, so it is the authoritative fallback when a checkpoint
# directory carries no copy of its own.
RUN_CONFIGS = {
    "v1.1": "configs/experiments/v1/v1.1.yaml",
    "v2.0-continued": "configs/experiments/v2/v2.0_continued.yaml",
    "v2.1-decmix": "configs/experiments/v2/v2.1_decmix.yaml",
    "v2.2-decmix-disc": "configs/experiments/v2/v2.2_decmix_disc.yaml",
    "v2.2-decmix-disc-old-asym": "configs/experiments/v2/v2.2_decmix_disc.yaml",
    "v2.3-encmix-g5": "configs/experiments/v2/v2.3_encmix_g5.yaml",
    "v2.4-encmix-g10": "configs/experiments/v2/v2.4_encmix_g10.yaml",
    "v2.5-encmix-g20": "configs/experiments/v2/v2.5_encmix_g20.yaml",
    "v2.6-decmix-frozenenc": "configs/experiments/v2/v2.6_decmix_frozenenc.yaml",
    "v2.7-mixedrecon": "configs/experiments/v2/v2.7_mixedrecon.yaml",
    "v3.0-baseline-d64": "configs/experiments/v3/v3.0_baseline_d64.yaml",
    "v3.1-decmix-disc-d64": "configs/experiments/v3/v3.1_decmix_disc_d64.yaml",
}


def resolve_config(name: str) -> Path | None:
    """Config for a run: the checkpoint dir's own copy, else the tracked one.

    A seed re-run ("<base>-s2") shares its base run's architecture.
    """
    local = REPO / "checkpoints" / name / "config.yaml"
    if local.exists():
        return local
    base = name
    while base:
        rel = RUN_CONFIGS.get(base)
        if rel and (REPO / rel).exists():
            return REPO / rel
        if "-s" in base and base.rsplit("-s", 1)[1].isdigit():
            base = base.rsplit("-s", 1)[0]
            continue
        return None
    return None


def out_dir(arg: str | None, default: Path) -> Path:
    """Where a driver writes. Kept separate so a smoke run cannot land on the
    published reference filenames: pass --out-dir results/smoke."""
    d = Path(arg) if arg else default
    if not d.is_absolute():
        d = REPO / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_common_args(ap, default_out: str):
    ap.add_argument("--out-dir", default=None,
                    help=f"Where to write results (default: {default_out}). "
                         "Use a separate directory for smoke or reproduction "
                         "runs so the published records stay untouched.")


def report(summary, requested_explicitly: bool) -> int:
    """Print evaluated/skipped/failed and return the process exit code."""
    ok = [n for n, s in summary if s == "ok"]
    skipped = [(n, s) for n, s in summary if s in ("exists",)]
    failed = [(n, s) for n, s in summary if s not in ("ok", "exists")]
    print(f"\nevaluated {len(ok)}, skipped {len(skipped)}, failed {len(failed)}")
    for n, s in summary:
        print(f"  {n:32s} {s}")
    if failed:
        print("\nfailed runs are listed above; missing weights come from "
              "https://huggingface.co/SoMa25/mixing-equivariant-ae-checkpoints")
        # A run the user named explicitly must not fail quietly.
        return 1 if requested_explicitly else 0
    return 0
