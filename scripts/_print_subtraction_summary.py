"""Print the subtraction summary tables across all completed runs."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "evaluation" / "v2_metrics"

NAMES = [
    "v2.0-continued",
    "v2.2-decmix-disc",
    "v3.1-decmix-disc-d64",
    "m2l-phase2",
]

for name in NAMES:
    fp = OUT / f"{name}_subtraction.json"
    if not fp.exists():
        print(f"\n{name}: (not yet eval'd)")
        continue
    p = json.load(open(fp))
    s = p["subtraction"]
    n = p["config"]["n_chunks_seen"]
    skipped = p.get("skipped_tracks", [])
    print(f"\n{name}  (n_chunks={n}, skipped={len(skipped)})")
    print(f"  {'stem':9s}  {'sdr_sub':>9s}  {'sdr_ceil':>9s}  {'gap':>7s}")
    for st in ["drums", "bass", "vocals", "other", "all"]:
        v = s[st]
        print(f"  {st:9s}  {v['sdr_sub']:+9.3f}  {v['sdr_ceil']:+9.3f}  {v['gap']:+7.3f}")
