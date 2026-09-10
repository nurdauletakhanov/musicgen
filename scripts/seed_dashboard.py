#!/usr/bin/env python3
"""Local dashboard for the ICASSP seed re-run pipeline on the cluster.

Same design as monitor_dashboard.py: ONE batched SSH call per interval into a
local cache, served on http://127.0.0.1:8788; the page refreshes from the
cache only, so browser reloads never touch the cluster.

Shows: pipeline stages (download -> preprocess -> chunks), each seed run's
step progress / speed / ETA / Slurm state, and the evaluation results as they
land next to the paper's original single-seed numbers (mean +/- std).

Usage:
    python3 scripts/seed_dashboard.py                  # port 8788, poll 5 min
    python3 scripts/seed_dashboard.py --interval 180 --port 8788
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Personal operations settings, all overridable so this is not tied to one
# machine or one deadline:
#   MUSICGEN_SSH_HOST, MUSICGEN_REMOTE_ROOT, MUSICGEN_DEADLINE (YYYY-MM-DD HH:MM)
SSH_HOST = os.environ.get("MUSICGEN_SSH_HOST", "lab-login")
REMOTE = os.environ.get("MUSICGEN_REMOTE_ROOT", "~/musicgen-seeds")
MAX_STEPS = 25000
DEADLINE = time.mktime(time.strptime(
    os.environ.get("MUSICGEN_DEADLINE", "2026-09-16 23:59"), "%Y-%m-%d %H:%M")) + 12 * 3600  # AoE
DECISION = time.mktime(time.strptime(
    os.environ.get("MUSICGEN_DECISION", "2026-09-12 18:00"), "%Y-%m-%d %H:%M"))

# base run -> (label, Slurm job-name prefix)
BASES = {
    "v2.0-continued":   ("baseline (no mix)", "v20"),
    "v2.1-decmix":      ("L_dec", "v21"),
    "v2.2-decmix-disc": ("L_dec + disc", "v22"),
}
SEEDS = ["s2", "s3"]
RUNS = [f"{b}-{s}" for b in BASES for s in SEEDS]
METRICS = [("sdr_lin_gt", "SDR_lin^gt", 2), ("mix_rate", "MixRate", 3),
           ("sdr_rec", "SDR_rec", 2), ("fad", "FAD", 4), ("sub_all", "Subtraction", 2)]

state = {"stages": {}, "runs": {}, "evals": {}, "queue": [], "launch": [],
         "last_poll": None, "last_error": None, "next_poll": None,
         "poll_count": 0, "interval": 300}
lock = threading.Lock()


def paper_values():
    """The paper's original single-seed numbers, from the local eval JSONs."""
    out = {}
    d = os.path.join(REPO, "evaluation", "v2_metrics")
    for b in BASES:
        v = {}
        try:
            m = json.load(open(os.path.join(d, f"{b}_mixing.json")))["metrics"]
            for k in ("sdr_lin_gt", "mix_rate", "sdr_rec"):
                v[k] = m[k]["all"]
        except Exception:
            pass
        try:
            v["fad"] = json.load(open(os.path.join(d, f"{b}_fad.json")))["fad/all"]
        except Exception:
            pass
        try:
            v["sub_all"] = json.load(open(os.path.join(d, f"{b}_subtraction.json")))["subtraction"]["all"]["sdr_sub"]
        except Exception:
            pass
        out[b] = v
    return out


PAPER = paper_values()


def remote_cmd():
    p = ["squeue -h -u $USER -o '%i|%j|%T|%M|%N|%R' 2>/dev/null",
         "echo '===PRE==='",
         f"grep -hE '^\\[|extracted|mp3s|wavs|Error|Traceback|train |test ' {REMOTE}/slurm-preprocess-*.out 2>/dev/null | tail -n 8",
         "echo '===CHUNKS==='",
         f"[ -f {REMOTE}/.chunks_ready ] && echo READY || echo pending",
         f"ls {REMOTE}/chunks-44k-1s/train 2>/dev/null | wc -l",
         f"ls {REMOTE}/chunks-44k-1s/test 2>/dev/null | wc -l",
         "echo '===DL==='", "[ -f ~/raw/.downloads_done ] && echo done || echo pending"]
    for r in RUNS:
        base = r.rsplit("-", 1)[0]; tag = BASES[base][1] + r.rsplit("-", 1)[1]
        p += [f"echo '===RUN {r}==='",
              f"tail -n 2 {REMOTE}/checkpoints/{r}/train_log.jsonl 2>/dev/null",
              f"[ -f {REMOTE}/checkpoints/{r}/step_25000.pth ] && echo FINAL",
              # per-step tqdm bar from the newest Slurm log (updates every step,
              # whereas train_log.jsonl only lands every log_every_steps=100)
              f"f=$(ls -t {REMOTE}/slurm-{tag}-*.out 2>/dev/null | head -1); "
              f"[ -n \"$f\" ] && tail -c 400 \"$f\" | tr '\\r' '\\n' | grep -oE '[0-9]+/25000 \\[[^]]*\\]' | tail -1 | sed 's/^/BAR /'"]
    p += ["echo '===EVALS==='",
          f"cd {REMOTE}/evaluation/v2_metrics 2>/dev/null && for f in *-s[0-9]_mixing.json *-s[0-9]_fad.json *-s[0-9]_subtraction.json; do [ -f \"$f\" ] && echo \"===JSON $f===\" && cat \"$f\"; done",
          "echo '===LAUNCH==='", "tail -n 8 ~/seeds_launch.log 2>/dev/null", "true"]
    return " ; ".join(p)


def parse(text):
    sec, cur, buf = {}, "squeue", []
    for line in text.splitlines():
        m = re.match(r"===(\S+)(?: (\S+))?===$", line)
        if m:
            sec[cur] = buf; buf = []
            cur = m.group(1) + (" " + m.group(2) if m.group(2) else "")
        else:
            buf.append(line)
    sec[cur] = buf

    queue = []
    for l in sec.get("squeue", []):
        f = l.split("|")
        if len(f) >= 6:
            queue.append(dict(id=f[0], name=f[1], state=f[2], elapsed=f[3], node=f[4], reason=f[5]))

    ch = sec.get("CHUNKS", ["pending", "0", "0"]) + ["0", "0"]
    stages = {
        "download": {"done": (sec.get("DL", ["pending"])[0].strip() == "done")},
        "preprocess": {"log": [l for l in sec.get("PRE", []) if l.strip()][-6:],
                       "running": any(q["name"] == "preprocess" for q in queue),
                       "train_files": int(ch[1] or 0), "test_files": int(ch[2] or 0)},
        "chunks": {"ready": ch[0].strip() == "READY"},
    }

    runs = {}
    for r in RUNS:
        lines = sec.get(f"RUN {r}", [])
        last = None
        for l in lines:
            l = l.strip()
            if l.startswith("{"):
                try:
                    last = json.loads(l)
                except Exception:
                    pass
        final = any(l.strip() == "FINAL" for l in lines)
        base = r.rsplit("-", 1)[0]
        tag = BASES[base][1] + r.rsplit("-", 1)[1]
        job = next((q for q in queue if q["name"] == tag), None)
        step = last["step"] if last else 0
        sps = last.get("steps_per_sec") if last else None
        bar = next((l for l in lines if l.startswith("BAR ")), None)
        if bar:
            m = re.search(r"(\d+)/25000 \[.*?,\s*([\d.]+)(s/it|it/s)\]", bar)
            if m and int(m.group(1)) >= step:
                step = int(m.group(1))
                v = float(m.group(2))
                sps = (1.0 / v if m.group(3) == "s/it" else v) if v > 0 else sps
        eta_h = ((MAX_STEPS - step) / sps / 3600) if (sps and step < MAX_STEPS) else None
        # A run that reached MAX_STEPS is finished even without the step_25000.pth
        # marker: that file is written after a final validation pass, which is
        # an hour long and is the easiest thing to lose to a kill. latest.pth
        # already holds the step-25000 weights in that case.
        done = final or step >= MAX_STEPS
        status = ("finished" if done else
                  "running" if job and job["state"] == "RUNNING" else
                  "queued" if job else
                  "stalled" if step else "waiting")
        runs[r] = dict(name=r, base=base, seed=r.rsplit("-", 1)[1], label=BASES[base][0],
                       step=step, steps_per_sec=sps, eta_h=eta_h, status=status,
                       job=job, loss=(last.get("loss/total") if last else None))

    evals = {}
    for k, v in sec.items():
        if k.startswith("JSON "):
            fname = k[5:]
            try:
                d = json.loads("\n".join(v))
            except Exception:
                continue
            run = re.sub(r"_(mixing|fad|subtraction)\.json$", "", fname)
            e = evals.setdefault(run, {})
            if fname.endswith("_mixing.json"):
                for m in ("sdr_lin_gt", "mix_rate", "sdr_rec"):
                    e[m] = d["metrics"][m]["all"]
            elif fname.endswith("_fad.json"):
                e["fad"] = d.get("fad/all")
            else:
                e["sub_all"] = d["subtraction"]["all"]["sdr_sub"]

    launch = [l for l in sec.get("LAUNCH", []) if l.strip()]
    return stages, runs, evals, queue, launch


def poll_loop(base_interval):
    fails = 0
    while True:
        try:
            out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                  SSH_HOST, remote_cmd()], capture_output=True, text=True, timeout=900)
            if out.returncode != 0 and not out.stdout:
                raise RuntimeError(out.stderr.strip()[-300:] or "ssh failed")
            stages, runs, evals, queue, launch = parse(out.stdout)
            with lock:
                state.update(stages=stages, runs=runs, evals=evals, queue=queue,
                             launch=launch, last_poll=time.time(), last_error=None)
                state["poll_count"] += 1
            fails = 0
        except Exception as e:
            fails += 1
            with lock:
                state["last_error"] = str(e)[:300]
        interval = base_interval * min(2 ** fails, 6)
        with lock:
            state["interval"] = interval
            state["next_poll"] = time.time() + interval
        time.sleep(interval)


def summary():
    """Aggregate per base: paper value, seed values, mean +/- std over all available runs."""
    with lock:
        evals = dict(state["evals"])
    rows = []
    for b, (label, _) in BASES.items():
        row = {"base": b, "label": label, "metrics": {}}
        for key, mlabel, nd in METRICS:
            vals = []
            if PAPER.get(b, {}).get(key) is not None:
                vals.append(PAPER[b][key])
            seeds = {s: evals.get(f"{b}-{s}", {}).get(key) for s in SEEDS}
            vals += [v for v in seeds.values() if v is not None]
            row["metrics"][key] = {
                "label": mlabel, "nd": nd, "paper": PAPER.get(b, {}).get(key),
                "seeds": seeds, "n": len(vals),
                "mean": statistics.mean(vals) if vals else None,
                "std": statistics.stdev(vals) if len(vals) > 1 else None}
        rows.append(row)
    return rows


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>ICASSP seeds</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { color-scheme: light; --surface:#fcfcfb; --surface-2:#f0efec; --border:#dddcd8;
  --text-1:#0b0b0b; --text-2:#52514e; --text-3:#8a8985; --s1:#2a78d6; --s2:#eb6834;
  --s3:#1baf7a; --good:#008300; --warn:#c98500; --bad:#e34948; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --surface:#1a1a19;
  --surface-2:#262625; --border:#3a3a38; --text-1:#fff; --text-2:#c3c2b7; --text-3:#8a8985;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --good:#4bb54b; --warn:#e0a020; --bad:#f06060; } }
* { box-sizing:border-box; margin:0; }
body { background:var(--surface); color:var(--text-1); font:14px/1.45 -apple-system,"Segoe UI",sans-serif;
  padding:24px; max-width:1060px; margin:0 auto; }
h1 { font-size:19px; margin-bottom:2px; } h2 { font-size:14px; margin:20px 0 8px; }
.sub { color:var(--text-2); margin-bottom:18px; font-size:13px; } .sub .err { color:var(--bad); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }
.tile { background:var(--surface-2); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
.tile h3 { font-size:13px; font-weight:600; display:flex; gap:8px; align-items:center; margin-bottom:6px; }
.dot { width:10px; height:10px; border-radius:3px; display:inline-block; }
.big { font-size:24px; font-weight:650; letter-spacing:-.02em; } .big small { font-size:13px; color:var(--text-2); font-weight:400; }
.meta { color:var(--text-2); font-size:12px; margin-top:4px; }
.bar { height:5px; background:var(--border); border-radius:3px; margin-top:9px; overflow:hidden; } .bar i { display:block; height:100%; }
.status { font-size:12px; padding:1px 8px; border-radius:999px; border:1px solid var(--border); color:var(--text-2); margin-left:auto; }
.status.running,.status.finished,.status.done { color:var(--good); border-color:var(--good); }
.status.queued,.status.waiting { color:var(--warn); border-color:var(--warn); }
.status.stalled,.status.pending { color:var(--bad); border-color:var(--bad); }
table { border-collapse:collapse; font-size:12.5px; font-variant-numeric:tabular-nums; width:100%; background:var(--surface-2);
  border:1px solid var(--border); border-radius:10px; overflow:hidden; }
th,td { padding:6px 10px; text-align:right; border-bottom:1px solid var(--border); } th { color:var(--text-2); font-weight:600; }
td:first-child,th:first-child { text-align:left; } tr:last-child td { border-bottom:none; }
.ms { font-weight:650; } .n { color:var(--text-3); }
pre { background:var(--surface-2); border:1px solid var(--border); border-radius:10px; padding:10px 12px; font-size:12px;
  color:var(--text-2); overflow-x:auto; white-space:pre-wrap; }
.count { display:flex; gap:22px; margin:6px 0 16px; font-size:13px; color:var(--text-2); } .count b { color:var(--text-1); }
</style></head><body>
<h1>ICASSP 2027 seed re-runs</h1>
<div class="sub" id="sub">loading…</div>
<div class="count" id="count"></div>
<h2>Pipeline</h2><div class="tiles" id="stages"></div>
<h2>Seed runs (25k steps each, 2 concurrent under the cluster quota)</h2><div class="tiles" id="runs"></div>
<h2>Results vs the paper's single-seed numbers (α=0.5, full test set)</h2><div id="results"></div>
<h2>Slurm queue</h2><div id="queue"></div>
<h2>Launcher log</h2><pre id="launch"></pre>
<script>
const C = {"v2.0-continued":"var(--s1)","v2.1-decmix":"var(--s2)","v2.2-decmix-disc":"var(--s3)"};
const ago = ts => !ts ? "never" : (s => s<90 ? s+"s ago" : Math.round(s/60)+" min ago")(Math.round(Date.now()/1000-ts));
const dhm = h => h==null ? "—" : (h<1 ? Math.round(h*60)+" min" : h.toFixed(1)+" h");
function tile(title, status, big, small, meta, pct, color) {
  return `<div class="tile"><h3>${color?`<i class="dot" style="background:${color}"></i>`:""}${title}
    <span class="status ${status}">${status}</span></h3><div class="big">${big}<small> ${small||""}</small></div>
    <div class="meta">${meta||""}</div>${pct!=null?`<div class="bar"><i style="width:${pct.toFixed(1)}%;background:${color||"var(--good)"}"></i></div>`:""}</div>`;
}
async function refresh(){
  let d; try { d = await (await fetch("/status.json")).json(); } catch(e){ document.getElementById("sub").textContent="dashboard server unreachable"; return; }
  const now = Date.now()/1000;
  document.getElementById("sub").innerHTML = `cluster polled ${ago(d.last_poll)} · next poll in ~${Math.max(0,Math.round((d.next_poll-now)/60))} min · poll #${d.poll_count}` + (d.last_error?` · <span class="err">last poll failed: ${d.last_error}</span>`:"");
  const days = s => (s/86400).toFixed(1);
  document.getElementById("count").innerHTML = `<span>decision point Sep 12: <b>${days(d.decision-now)} days</b></span><span>ICASSP deadline Sep 16 AoE: <b>${days(d.deadline-now)} days</b></span>`;
  const st = d.stages || {}, pre = st.preprocess||{}, ch = st.chunks||{};
  const preStatus = ch.ready ? "done" : (pre.running ? "running" : (st.download&&st.download.done ? "pending" : "waiting"));
  document.getElementById("stages").innerHTML =
    tile("Raw datasets", st.download&&st.download.done?"done":"pending", st.download&&st.download.done?"209 GB":"…", "verified", "FMA-large, MAESTRO v3, MUSDB18-HQ") +
    tile("Preprocess → chunks-44k-1s", preStatus, (pre.train_files||0).toLocaleString(), "train files", (pre.log||[]).slice(-1)[0]||"", null) +
    tile("Chunks ready", ch.ready?"done":"pending", ch.ready?"yes":"no", "", ch.ready?"seed launcher fires automatically":"launcher waiting", null);
  const runs = Object.values(d.runs||{});
  document.getElementById("runs").innerHTML = runs.map(r => {
    const pct = 100*r.step/25000, job = r.job ? ` · ${r.job.node||r.job.reason} · ${r.job.elapsed}` : "";
    return tile(`${r.label} · ${r.seed}`, r.status, r.step.toLocaleString(), "/ 25,000 steps",
      `${r.steps_per_sec?r.steps_per_sec.toFixed(3)+" steps/s · ETA "+dhm(r.eta_h):"no log yet"}${job}`, pct, C[r.base]);
  }).join("");
  const rows = d.summary||[];
  let h = "<table><tr><th>config</th>" + Object.values(rows[0]?rows[0].metrics:{}).map(m=>`<th>${m.label}</th>`).join("") + "</tr>";
  for (const r of rows) { h += `<tr><td><i class="dot" style="background:${C[r.base]}"></i> ${r.label}</td>`;
    for (const m of Object.values(r.metrics)) {
      const f = v => v==null?"—":v.toFixed(m.nd);
      const seeds = Object.entries(m.seeds).map(([s,v])=>`${s} ${f(v)}`).join(" · ");
      h += `<td><span class="ms">${f(m.mean)}${m.std!=null?" ± "+m.std.toFixed(m.nd):""}</span> <span class="n">n=${m.n}</span><br><span class="n">paper ${f(m.paper)} · ${seeds}</span></td>`; }
    h += "</tr>"; }
  document.getElementById("results").innerHTML = h + "</table>";
  const q = d.queue||[];
  document.getElementById("queue").innerHTML = q.length ? "<table><tr><th>job</th><th>name</th><th>state</th><th>elapsed</th><th>node / reason</th></tr>" +
    q.map(j=>`<tr><td>${j.id}</td><td>${j.name}</td><td>${j.state}</td><td>${j.elapsed}</td><td>${j.node||j.reason}</td></tr>`).join("") + "</table>" : "<div class='meta'>no jobs in queue</div>";
  document.getElementById("launch").textContent = (d.launch||[]).join("\n") || "(nothing launched yet)";
}
refresh(); setInterval(refresh, 60000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/status.json":
            with lock:
                d = dict(state)
            d["summary"] = summary()
            d["deadline"] = DEADLINE
            d["decision"] = DECISION
            body = json.dumps(d).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()
    threading.Thread(target=poll_loop, args=(args.interval,), daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"seed dashboard: http://{args.host}:{args.port}  (pid {os.getpid()})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
