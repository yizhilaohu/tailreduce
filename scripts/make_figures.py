"""Publication figures from the experiment results.

Figure 2 (fig_regret_vs_K.pdf): worst-case regret / M = D+ + D- versus budget K (E1, N = 5000).
Figure 3 (fig_kmin_vs_a.pdf):   minimum exact size K_min / N versus tail mass a under uniform and
                                time-decayed weights (E2, N = 2000).

Color follows the method (fixed categorical slots, validated for adjacent-pair CVD separation);
markers and line styles give a second, color-independent encoding for print.
Usage: python scripts/make_figures.py [--outdir ../paper/figures]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# fixed categorical slots (reference palette, validated): entity -> (color, marker, linestyle, label)
STYLE = {
    "OPT": ("#2a78d6", "o", "-", r"OPT $=$ OPT-C $=$ TB"),
    "W2": ("#eb6834", "s", "-", "W2: $K$-means"),
    "W1": ("#1baf7a", "D", "-", "W1: $K$-medians"),
    "QM": ("#eda100", "v", "-", "QM: quantile medians"),
    "RS": ("#e87ba4", "^", "-", "RS: random sampling"),
    "KW": ("#008300", "x", "-", "KW: keep $K$ worst"),
    "TW2": ("#6250d6", "P", "-", "TW2: tail $K$-means"),
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
GUIDE = "#8a8984"
BLUE_RAMP = {"uniform": "#104281", 1000: "#256abf", 250: "#5598e7", 63: "#9ec5f4"}


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": GUIDE,
        "axes.linewidth": 0.5,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def fig_regret(outdir):
    df = pd.read_csv(ROOT / "results" / "e1_worst_case.csv")
    df = df[df.N == 5000]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for ax, a in zip(axes, (0.01, 0.05, 0.10)):
        sub = df[df.a == a]
        n_a = int(sub.n_a.iloc[0])
        Ks = np.array(sorted(sub.K.unique()))
        kk = np.logspace(np.log10(Ks.min()), np.log10(Ks.max()), 100)
        ax.plot(kk, np.minimum(1, 1 / kk), color=GUIDE, lw=0.8, ls=(0, (4, 2)), zorder=1)
        if n_a <= 240:
            ax.axvline(n_a, color=INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=4)
            ax.text(n_a * 0.92, 0.0028, rf"$K_{{\min}}={n_a}$", color=INK2, fontsize=6.5, rotation=90,
                    va="bottom", ha="right")
        else:
            ax.text(2.1, 0.0028, rf"$K_{{\min}}={n_a}$ (off scale)", color=INK2, fontsize=6.5, ha="left", va="bottom")
        for m in ("RS", "QM", "KW", "W1", "W2", "TW2", "OPT"):
            g = sub[sub.method == m].groupby("K").D.mean().reindex(Ks)
            y = g.to_numpy()
            mask = y > 1e-9  # exact models (zero) for K >= K_min are not drawn on the log scale
            c, mk, ls, lab = STYLE[m]
            ax.plot(Ks[mask], y[mask], color=c, marker=mk, ls=ls, label=lab,
                    markeredgecolor="white" if mk != "x" else c, markeredgewidth=0.5 if mk != "x" else 1.0,
                    zorder=3 if m == "OPT" else 2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.0025, 1.4)
        ax.set_xlim(1.7, 240)
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
        ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
        ax.grid(True, which="major", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(rf"tail mass $a={a:g}$", color=INK)
        ax.set_xlabel("budget $K$ (scenarios kept)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"worst-case regret / $M$  $=D^+ + D^-$")
    axes[0].text(62, 0.021, r"$1/K$", color=INK2, fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(STYLE[m][3]) for m in ("OPT", "TW2", "W2", "W1", "QM", "RS", "KW")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.09), handlelength=1.8, columnspacing=0.9, fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(outdir / "fig_regret_vs_K.pdf")
    fig.savefig(outdir / "fig_regret_vs_K.png", dpi=200)
    plt.close(fig)


def fig_kmin(outdir):
    cur = pd.read_csv(ROOT / "results" / "e2_kmin_curve.csv")
    fig, ax = plt.subplots(figsize=(3.35, 1.85))
    labels = {"uniform": "equal weights", "1000": "half-life 1000 days", "250": "half-life 250 days",
              "63": "half-life 63 days"}
    styles = {"uniform": (0, (1, 0)), "1000": (0, (5, 1.5)), "250": (0, (2.5, 1.5)), "63": (0, (1, 1.2))}
    for hl in ("63", "250", "1000", "uniform"):
        sub = cur[cur.half_life.astype(str) == hl]
        g = sub.groupby("a").kmin_ratio
        a = np.array(sorted(sub.a.unique()))
        med = g.median().reindex(a).to_numpy()
        lo = g.quantile(0.10).reindex(a).to_numpy()
        hi = g.quantile(0.90).reindex(a).to_numpy()
        key = "uniform" if hl == "uniform" else int(hl)
        c = BLUE_RAMP[key]
        if hl in ("63", "250"):
            ax.fill_between(a, lo, hi, color=c, alpha=0.22 if hl == "63" else 0.30, lw=0,
                            label=f"10--90% range, {hl} days")
        ax.plot(a, med, color=c if hl != "63" else "#5598e7", lw=1.4, ls=styles[hl], label=f"median, {labels[hl]}"
                if hl != "uniform" else labels[hl])
    ax.set_xlabel("tail mass $a$")
    ax.set_ylabel(r"$K_{\min}/N$")
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 0.3)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    want = ["equal weights", "median, half-life 1000 days", "median, half-life 250 days", "median, half-life 63 days",
            "10--90% range, 250 days", "10--90% range, 63 days"]
    idx = [l.index(x) for x in want if x in l]
    ax.legend([h[i] for i in idx], [l[i] for i in idx], frameon=False, loc="upper left", handlelength=2.0,
              fontsize=6.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_kmin_vs_a.pdf")
    fig.savefig(outdir / "fig_kmin_vs_a.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT.parent / "paper" / "figures"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    setup()
    fig_regret(out)
    fig_kmin(out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
