"""Generate paper figures from the eval JSONs / diagnostic.

Outputs (research/paper/icassp2027/figures/):
  fig_alpha.pdf    — SI-SDR_lin_gt vs mixing coefficient alpha (mixing flattens
                     the equivariance curve). Source: alpha_sweep_summary.json.
  fig_origin.pdf   — latent-origin intervention: raw vs recentered stem
                     subtraction per model (left) and the per-track change in
                     between-model advantage (right).
  fig_protocol.pdf — decode-noise protocol confound on Music2Latent: the same
                     checkpoint scores deeply negative under independent-noise
                     decoding and positive under shared-noise decoding.
                     Source: scripts/_diag_old_vs_new_eval (240 MUSDB chunks).

Usage:
  python research/paper/icassp2027/make_figures.py
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
M = os.path.join(REPO, "evaluation", "v2_metrics")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    # ICASSP requires >= 9 pt throughout, so no tick/legend text below 9.
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 200,
    # IEEE PDF eXpress rejects Type 3 fonts (matplotlib's PDF default).
    # Type 42 embeds the glyphs as TrueType instead.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    # Match the IEEEtran body font so figure text blends with the page.
    "font.family": "serif", "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def fig_alpha():
    summ = json.load(open(os.path.join(M, "alpha_sweep_summary.json")))
    # Models to show: no-mix baseline vs mixing variants (GAN AE).
    series = [
        ("v2.0-continued",   "no mixing (v2.0)",        "o", "--"),
        ("v2.1-decmix",      "$\\mathcal{L}_\\mathrm{dec}$ (v2.1)", "s", "-"),
        ("v2.2-decmix-disc", "$\\mathcal{L}_\\mathrm{dec}$+disc (v2.2)", "^", "-"),
    ]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for name, label, mk, ls in series:
        if name not in summ:
            continue
        xs, ys = [], []
        for a in ALPHAS:
            cell = summ[name].get(str(a))
            if cell and cell.get("sdr_lin_gt") is not None:
                xs.append(a)
                ys.append(cell["sdr_lin_gt"])
        ax.plot(xs, ys, marker=mk, linestyle=ls, label=label, markersize=4)
    ax.set_xlabel(r"mixing coefficient $\alpha$")
    ax.set_ylabel(r"SI-SDR$_\mathrm{lin}^\mathrm{gt}$ (dB)")
    ax.set_xticks(ALPHAS)
    ax.grid(True, alpha=0.3)
    # Legend above the axes: the curves dip at alpha=0.5, right where an
    # in-axes legend would sit.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_alpha.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {os.path.relpath(out, REPO)}")


def fig_protocol():
    """Decode-noise protocol confound. Parsed from the diagnostic log so the
    figure cannot drift from the run (scripts/slurm_fig1_rerun.sh, 240 MUSDB
    chunks, alpha=0.5, identical pairs across protocols)."""
    log = open(os.path.join(M, "_diag_old_vs_new_eval_fixedpairs.log")).read()
    vals = {}
    for a, b, _rec, lin in re.findall(
            r"^(shared|random)\s+(P[02])\s+([-+0-9.]+)\s+([-+0-9.]+)$", log, re.M):
        vals[(a, b)] = float(lin)
    data = {
        "shared decode noise":     {"M2L base": vals[("shared", "P0")],
                                               "M2L +mix": vals[("shared", "P2")]},
        "independent decode noise": {"M2L base": vals[("random", "P0")],
                                               "M2L +mix": vals[("random", "P2")]},
    }
    models = ["M2L base", "M2L +mix"]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    import numpy as np
    x = np.arange(len(models))
    w = 0.38
    for i, (proto, v) in enumerate(data.items()):
        ax.bar(x + (i - 0.5) * w, [v[m] for m in models], w, label=proto)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(r"SI-SDR$_\mathrm{lin}$ (dB)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, columnspacing=1.2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_protocol.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {os.path.relpath(out, REPO)}  " +
          ", ".join(f"{k[0]}/{k[1]}={v:+.2f}" for k, v in sorted(vals.items())))


# Order and display names for the five GAN checkpoints in the origin figure.
ORIGIN_ROWS = [
    ("v2.0-continued",        "7.66$\\times$ no mix"),
    ("v2.1-decmix",           "7.66$\\times$ $+\\mathcal{L}_\\mathrm{dec}$"),
    ("v2.2-decmix-disc",      "7.66$\\times$ $+\\mathcal{L}_\\mathrm{dec}$+disc"),
    ("v3.0-baseline-d64",     "15.3$\\times$ no mix"),
    ("v3.1-decmix-disc-d64",  "15.3$\\times$ $+\\mathcal{L}_\\mathrm{dec}$+disc"),
]


def fig_origin():
    """Recentering the origin: every model lands on its own reconstruction
    reference, and the apparent advantage of mixing supervision disappears."""
    d = json.load(open(os.path.join(M, "origin_effect.json")))
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.9, 1.88),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    ys = list(range(len(ORIGIN_ROWS)))[::-1]
    for y, (key, label) in zip(ys, ORIGIN_ROWS):
        m = d["models"][key]
        raw, cor, ref = m["sub"], d["models"][key + "+origin"]["sub"], m["reference"]
        axl.annotate("", xy=(cor, y), xytext=(raw, y),
                     arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.0,
                                     shrinkA=2.5, shrinkB=3.0))
        axl.plot([raw], [y], "o", ms=4.5, mfc="white", mec="0.25", mew=1.0, zorder=3)
        axl.plot([cor], [y], "o", ms=4.5, color="#1f4e79", zorder=3)
        axl.plot([ref], [y], "|", ms=11, mew=1.4, color="#b03a2e", zorder=4)
    axl.set_yticks(ys[::-1])
    axl.set_yticklabels([lab for _, lab in ORIGIN_ROWS][::-1])
    axl.set_ylim(-0.7, len(ORIGIN_ROWS) - 0.3)
    axl.set_xlabel("stem-subtraction SI-SDR (dB)")
    axl.grid(True, axis="x", alpha=0.3)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec="0.25", label="raw"),
          plt.Line2D([], [], marker="o", ls="", ms=4.5, color="#1f4e79", label="recentered"),
          plt.Line2D([], [], marker="|", ls="", ms=9, mew=1.4, color="#b03a2e", label="reference")]
    axl.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False, columnspacing=1.0, handletextpad=0.4)

    marks = ["o", "s", "^"]
    cols = ["#1f4e79", "#b03a2e", "#4a7c2f"]
    for (k, pair), mk, c in zip(sorted(d["pairs"].items()), marks, cols):
        axr.scatter(pair["per_track_raw"], pair["per_track_corrected"], s=9,
                    marker=mk, facecolor="none", edgecolor=c, linewidth=0.7,
                    label=pair["label"].replace("7.66x", "7.66$\\times$").replace("15.3x", "15.3$\\times$"))
    lim = [-1.5, 7.0]
    axr.plot(lim, lim, "-", color="0.6", lw=0.8, zorder=0)
    axr.axhline(0.0, color="0.3", lw=0.8, ls=":", zorder=0)
    axr.set_xlim(*lim); axr.set_ylim(-1.6, 3.2)
    axr.set_xlabel("raw advantage over control (dB)")
    axr.set_ylabel("after recentering (dB)")
    axr.grid(True, alpha=0.3)
    axr.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1,
               frameon=False, handletextpad=0.3, labelspacing=0.2, borderpad=0.1)

    fig.tight_layout(pad=0.3, w_pad=1.4)
    out = os.path.join(FIG, "fig_origin.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    fig_alpha()
    fig_origin()
    fig_protocol()
