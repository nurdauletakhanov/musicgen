#!/usr/bin/env python3
"""Render a shareable snapshot of the seed-run dashboard as a standalone page.

The live dashboard (scripts/seed_dashboard.py) binds to localhost, so it is not
reachable from a phone. This takes the same status JSON and writes a
self-contained HTML file that can be published as an Artifact and opened from
any device. Re-run and re-publish to refresh.

Usage:
    python3 scripts/seed_snapshot.py                       # from the live dashboard
    python3 scripts/seed_snapshot.py --status /tmp/s.json  # from a saved poll
    python3 scripts/seed_snapshot.py --out /tmp/seeds.html
"""

import argparse
import html
import json
import time
import urllib.request

SERIES = {
    "v2.0-continued": "var(--s-base)",
    "v2.1-decmix": "var(--s-dec)",
    "v2.2-decmix-disc": "var(--s-disc)",
}
MAX_STEPS = 25000


def esc(x):
    return html.escape(str(x))


def fmt(v, nd):
    return "—" if v is None else f"{v:.{nd}f}"


def rel(ts, now):
    if not ts:
        return "unknown"
    s = int(now - ts)
    return f"{s}s ago" if s < 90 else f"{s // 60} min ago"


def days(ts, now):
    return (ts - now) / 86400


def build(d, now):
    st = d.get("stages", {})
    pre = st.get("preprocess", {})
    ch = st.get("chunks", {})
    runs = list(d.get("runs", {}).values())
    queue = d.get("queue", [])
    summary = d.get("summary", [])

    finished = sum(1 for r in runs if r["status"] == "finished")
    running = sum(1 for r in runs if r["status"] == "running")
    evaluated = sum(1 for r in summary for m in [r["metrics"]["sdr_lin_gt"]] if m["n"] > 1)

    # Verdict: the one question this page exists to answer.
    d_dec = days(d.get("decision", now), now)
    if evaluated:
        verdict = f"{evaluated} of 3 configurations have error bars"
        vsub = "Table I can report mean ± std for these rows."
        vstate = "good"
    elif finished:
        verdict = f"{finished} of 6 runs finished, evaluations pending"
        vsub = f"{d_dec:.1f} days to the submission decision."
        vstate = "warn"
    elif running:
        verdict = f"{running} seed run{'s' if running != 1 else ''} training"
        vsub = f"First results expected ~24 h after launch · {d_dec:.1f} days to decide."
        vstate = "warn"
    elif ch.get("ready"):
        verdict = "Training data ready, seeds launching"
        vsub = f"{d_dec:.1f} days to the submission decision."
        vstate = "warn"
    else:
        verdict = "Rebuilding the training set"
        vsub = (f"Seeds start automatically when it finishes · "
                f"{d_dec:.1f} days to the submission decision.")
        vstate = "wait"

    # Pipeline stages, with the real counts that show it is actually moving.
    dl = st.get("download", {}).get("done")
    stages = [
        ("Raw corpora", "done" if dl else "wait",
         "209 GB verified" if dl else "downloading",
         "FMA-large · MAESTRO v3 · MUSDB18-HQ"),
        ("Chunking to 1 s @ 44.1 kHz",
         "done" if ch.get("ready") else ("run" if pre.get("running") else "wait"),
         f"{pre.get('train_files', 0):,} train shards",
         (pre.get("log") or ["queued"])[-1]),
        ("Seed training", "run" if running else ("done" if finished == 6 else "wait"),
         f"{finished}/6 runs", "2 concurrent (cluster quota)"),
    ]

    stage_html = "".join(
        f'<li class="stage {s[1]}"><span class="dot"></span>'
        f'<div><h3>{esc(s[0])}</h3><p class="fig">{esc(s[2])}</p>'
        f'<p class="note">{esc(s[3])}</p></div></li>' for s in stages)

    run_html = ""
    for r in runs:
        pct = 100 * r["step"] / MAX_STEPS
        eta = r.get("eta_h")
        detail = (f"{r['steps_per_sec']:.3f} steps/s · ETA "
                  f"{eta:.1f} h" if r.get("steps_per_sec") and eta else "not started")
        job = r.get("job")
        where = f" · {esc(job['node'] or job['reason'])}" if job else ""
        run_html += (
            f'<li class="run"><div class="run-head">'
            f'<span class="swatch" style="background:{SERIES[r["base"]]}"></span>'
            f'<span class="run-name">{esc(r["label"])}<span class="seed">{esc(r["seed"])}</span></span>'
            f'<span class="pill {esc(r["status"])}">{esc(r["status"])}</span></div>'
            f'<div class="track"><i style="width:{pct:.1f}%;background:{SERIES[r["base"]]}"></i></div>'
            f'<p class="note"><b>{r["step"]:,}</b> / {MAX_STEPS:,} steps · {esc(detail)}{where}</p>'
            f'</li>')

    res_html = ""
    for row in summary:
        rows = ""
        for key, m in row["metrics"].items():
            seeds = " · ".join(f'{s} {fmt(v, m["nd"])}' for s, v in m["seeds"].items())
            agg = fmt(m["mean"], m["nd"])
            if m["std"] is not None:
                agg += f' <span class="pm">± {m["std"]:.{m["nd"]}f}</span>'
            rows += (f'<tr><th>{esc(m["label"])}</th>'
                     f'<td class="num">{fmt(m["paper"], m["nd"])}</td>'
                     f'<td class="num seedcell">{esc(seeds)}</td>'
                     f'<td class="num agg">{agg}<span class="n">n={m["n"]}</span></td></tr>')
        res_html += (
            f'<section class="config"><h3><span class="swatch" '
            f'style="background:{SERIES[row["base"]]}"></span>{esc(row["label"])}'
            f'<span class="run-id">{esc(row["base"])}</span></h3>'
            f'<div class="tablewrap"><table><thead><tr><th>metric</th>'
            f'<th class="num">paper</th><th class="num">re-runs</th>'
            f'<th class="num">combined</th></tr></thead><tbody>{rows}</tbody></table></div>'
            f'</section>')

    q_html = "".join(
        f'<tr><td class="mono">{esc(j["id"])}</td><td>{esc(j["name"])}</td>'
        f'<td>{esc(j["state"])}</td><td class="mono">{esc(j["elapsed"])}</td>'
        f'<td class="mono">{esc(j["node"] or j["reason"])}</td></tr>' for j in queue)
    if not q_html:
        q_html = '<tr><td colspan="5" class="note">no jobs queued</td></tr>'

    stamp = time.strftime("%d %b, %H:%M", time.localtime(d.get("last_poll") or now))
    d_dl = days(d.get("deadline", now), now)

    return f"""<title>ICASSP Seed Runs</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --ground:#f5f7fa; --panel:#ffffff; --sunk:#eef2f7; --rule:#d9e1ea;
  --ink:#0e141b; --ink-2:#4c5764; --ink-3:#7e8a98;
  --s-base:#2f6fd0; --s-dec:#c8721c; --s-disc:#0f8474;
  --good:#137a45; --warn:#a8720a; --wait:#7e8a98;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0d1117; --panel:#161c24; --sunk:#11161d; --rule:#262e39;
    --ink:#e7ecf3; --ink-2:#9aa6b4; --ink-3:#6d7885;
    --s-base:#5b9bf0; --s-dec:#e09140; --s-disc:#2ba795;
    --good:#3fb573; --warn:#d9a13c; --wait:#6d7885;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0d1117; --panel:#161c24; --sunk:#11161d; --rule:#262e39;
  --ink:#e7ecf3; --ink-2:#9aa6b4; --ink-3:#6d7885;
  --s-base:#5b9bf0; --s-dec:#e09140; --s-disc:#2ba795;
  --good:#3fb573; --warn:#d9a13c; --wait:#6d7885;
}}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--ground); color:var(--ink);
  font:400 15px/1.5 "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  padding:20px 16px 56px; }}
main {{ max-width:660px; margin:0 auto; display:flex; flex-direction:column; gap:26px; }}
.mono, .num, .fig, .track + .note b {{ font-family:"IBM Plex Mono", ui-monospace, monospace; }}
header h1 {{ font-size:19px; font-weight:600; letter-spacing:-.01em; text-wrap:balance; }}
header p {{ color:var(--ink-3); font-size:12.5px; margin-top:3px;
  font-family:"IBM Plex Mono", ui-monospace, monospace; }}
.verdict {{ border-left:3px solid var(--wait); padding:12px 0 12px 14px; }}
.verdict.good {{ border-color:var(--good); }}
.verdict.warn {{ border-color:var(--warn); }}
.verdict h2 {{ font-size:17px; font-weight:600; letter-spacing:-.01em; text-wrap:balance; }}
.verdict p {{ color:var(--ink-2); font-size:13.5px; margin-top:3px; }}
.clocks {{ display:flex; gap:10px; }}
.clock {{ flex:1; background:var(--sunk); border:1px solid var(--rule);
  border-radius:8px; padding:10px 12px; }}
.clock b {{ display:block; font:600 20px/1.2 "IBM Plex Mono", monospace;
  font-variant-numeric:tabular-nums; }}
.clock span {{ color:var(--ink-3); font-size:11.5px; text-transform:uppercase;
  letter-spacing:.06em; }}
h2.sec {{ font-size:12px; text-transform:uppercase; letter-spacing:.08em;
  color:var(--ink-3); font-weight:600; }}
ul {{ list-style:none; padding:0; display:flex; flex-direction:column; gap:2px; }}
.stage {{ display:flex; gap:12px; padding:11px 0; border-bottom:1px solid var(--rule); }}
.stage:last-child {{ border-bottom:none; }}
.stage .dot {{ width:9px; height:9px; border-radius:50%; margin-top:6px; flex:none;
  background:var(--wait); }}
.stage.done .dot {{ background:var(--good); }}
.stage.run .dot {{ background:var(--warn); box-shadow:0 0 0 3px color-mix(in srgb, var(--warn) 22%, transparent); }}
.stage h3 {{ font-size:14px; font-weight:500; }}
.fig {{ font-size:13px; font-weight:500; margin-top:1px; font-variant-numeric:tabular-nums; }}
.note {{ color:var(--ink-3); font-size:12px; margin-top:2px; }}
.run {{ background:var(--panel); border:1px solid var(--rule); border-radius:9px;
  padding:11px 13px; }}
ul.runs {{ gap:8px; }}
.run-head {{ display:flex; align-items:center; gap:8px; }}
.swatch {{ width:9px; height:9px; border-radius:2px; flex:none; }}
.run-name {{ font-size:13.5px; font-weight:500; }}
.seed {{ color:var(--ink-3); font-family:"IBM Plex Mono", monospace;
  font-size:11.5px; margin-left:6px; }}
.pill {{ margin-left:auto; font-size:11px; letter-spacing:.03em; padding:1px 8px;
  border-radius:999px; border:1px solid var(--rule); color:var(--ink-3); }}
.pill.running {{ color:var(--warn); border-color:var(--warn); }}
.pill.finished {{ color:var(--good); border-color:var(--good); }}
.track {{ height:4px; background:var(--sunk); border-radius:2px; margin:9px 0 7px;
  overflow:hidden; }}
.track i {{ display:block; height:100%; border-radius:2px; }}
.config {{ margin-bottom:16px; }}
.config h3 {{ display:flex; align-items:center; gap:8px; font-size:14px;
  font-weight:600; margin-bottom:7px; }}
.run-id {{ margin-left:auto; color:var(--ink-3); font-size:11.5px;
  font-family:"IBM Plex Mono", monospace; }}
.tablewrap {{ overflow-x:auto; border:1px solid var(--rule); border-radius:9px;
  background:var(--panel); }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
th, td {{ padding:7px 11px; text-align:left; border-bottom:1px solid var(--rule);
  white-space:nowrap; }}
thead th {{ color:var(--ink-3); font-weight:500; font-size:11px;
  text-transform:uppercase; letter-spacing:.05em; }}
tbody th {{ font-weight:500; color:var(--ink-2); }}
tr:last-child td, tr:last-child th {{ border-bottom:none; }}
.num {{ text-align:right; font-family:"IBM Plex Mono", monospace;
  font-variant-numeric:tabular-nums; }}
.seedcell {{ color:var(--ink-3); }}
.agg {{ font-weight:600; }}
.pm {{ font-weight:400; color:var(--ink-2); }}
.n {{ color:var(--ink-3); font-weight:400; margin-left:6px; font-size:11px; }}
footer {{ color:var(--ink-3); font-size:12px; border-top:1px solid var(--rule);
  padding-top:14px; }}
footer code {{ font-family:"IBM Plex Mono", monospace; font-size:11.5px;
  background:var(--sunk); padding:1px 5px; border-radius:4px; }}
</style>
<main>
<header>
  <h1>Seed re-runs for the ICASSP submission</h1>
  <p>snapshot {esc(stamp)} · cluster polled {esc(rel(d.get('last_poll'), now))}</p>
</header>

<div class="verdict {vstate}">
  <h2>{esc(verdict)}</h2>
  <p>{esc(vsub)}</p>
</div>

<div class="clocks">
  <div class="clock"><b>{d_dec:.1f}</b><span>days to decide</span></div>
  <div class="clock"><b>{d_dl:.1f}</b><span>days to deadline</span></div>
</div>

<section>
  <h2 class="sec">Pipeline</h2>
  <ul>{stage_html}</ul>
</section>

<section>
  <h2 class="sec">Seed runs · 25,000 steps each</h2>
  <ul class="runs">{run_html}</ul>
</section>

<section>
  <h2 class="sec">Results vs the paper's single-seed numbers</h2>
  <p class="note" style="margin:0 0 12px">Full test set, α = 0.5. SI-SDR in dB.
  &ldquo;Combined&rdquo; is what Table I would report.</p>
  {res_html}
</section>

<section>
  <h2 class="sec">Cluster queue</h2>
  <div class="tablewrap"><table><thead><tr><th>job</th><th>name</th><th>state</th>
  <th>elapsed</th><th>node</th></tr></thead><tbody>{q_html}</tbody></table></div>
</section>

<footer>
  Static snapshot, not a live feed. Refresh by re-running
  <code>seed_snapshot.py</code> and re-publishing. The live dashboard is
  <code>127.0.0.1:8788</code> on the laptop.
</footer>
</main>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="http://127.0.0.1:8788/status.json")
    ap.add_argument("--out", default="/tmp/seed_snapshot.html")
    a = ap.parse_args()
    if a.status.startswith("http"):
        d = json.load(urllib.request.urlopen(a.status, timeout=10))
    else:
        d = json.load(open(a.status))
    open(a.out, "w").write(build(d, time.time()))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
