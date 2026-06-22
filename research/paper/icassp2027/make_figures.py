"""Generate paper figures from the eval JSONs / diagnostic.

Outputs (research/paper/icassp2027/figures/):
  fig_alpha.pdf    — SI-SDR_lin_gt vs mixing coefficient alpha (mixing flattens
                     the equivariance curve). Source: alpha_sweep_summary.json.
  fig_protocol.pdf — decode-noise protocol confound on Music2Latent: the same
                     checkpoint scores deeply negative under independent-noise
                     decoding and positive under shared-noise decoding.
                     Source: scripts/_diag_old_vs_new_eval (240 MUSDB chunks).

Usage:
  python research/paper/icassp2027/make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
M = os.path.join(REPO, "evaluation", "v2_metrics")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 200,
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
    ax.legend(loc="lower center", frameon=False, ncol=1)
    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_alpha.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {os.path.relpath(out, REPO)}")


def fig_protocol():
    # From scripts/_diag_old_vs_new_eval (240 MUSDB chunks, alpha=0.5).
    # SI-SDR_lin (decode-vs-decode) under the two decode-noise protocols.
    data = {
        "shared noise\n(phase cancelled)":     {"M2L base": 3.54, "M2L +mix": 5.96},
        "independent noise\n(prior protocol)": {"M2L base": -9.28, "M2L +mix": -7.62},
    }
    models = ["M2L base", "M2L +mix"]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    import numpy as np
    x = np.arange(len(models))
    w = 0.38
    for i, (proto, vals) in enumerate(data.items()):
        ax.bar(x + (i - 0.5) * w, [vals[m] for m in models], w, label=proto)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(r"SI-SDR$_\mathrm{lin}$ (dB)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_protocol.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    fig_alpha()
    fig_protocol()
