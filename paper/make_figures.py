"""Figures for the manuscript, drawn from the saved result files (no simulation rerun).

Usage: .venv/bin/python paper/make_figures.py   -> paper/figs/*.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figs")
PREVIEW = os.environ.get("FIG_PREVIEW")   # optional folder for PNG previews


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, name + ".png"), bbox_inches="tight", dpi=170)
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "legend.frameon": False})

LEST = "Load-aware Dijkstra, LEST load table (4 tiers)"
BATT = "Battery Dijkstra (reactive)"
LP2 = "Forecast LP v2, Holt-Winters (ours)"
EBR = "EBR-DA (Mahdi et al. 2018, isolated nodes bridged)"
MAD = "MADII (not retrained)"

dyn = json.load(open(os.path.join(ROOT, "results_dynamic.json")))
intel = json.load(open(os.path.join(ROOT, "results_intel_e1.json")))["L"]
static = json.load(open(os.path.join(ROOT, "results_compare.json")))
# the evaluated MADII rebuild (19.5 rounds) is in results_eval.json, on the same 30 deployments
static["MADII"] = {"pct_ceiling": json.load(open(os.path.join(ROOT, "results_eval.json")))["madii_v3 (informer, IL)"]["pct"]}
seeds = sorted(dyn["R"], key=int)
per = lambda m, s: np.array([dyn[s][k][m] for k in seeds])


def ladder():
    methods = [(MAD, MAD, "MADII (learned)", S1), (EBR, EBR, "EBR-DA", S2), (BATT, "Battery Dijkstra", "Battery-weighted Dijkstra", S3),
               (LEST, LEST, "LEST load-table Dijkstra (proposed)", S4), (LP2, LP2, "LP planner (reference, LP every round)", S5)]
    groups = ["Synthetic traffic\n(mean of R, N, B)", "Pattern shift D\n(not tuned on)", "Intel Lab traffic\n(1.0 J batteries)"]
    vals = [[np.mean([per(d, s).mean() for s in "RNB"]), per(d, "D").mean(), np.mean(intel[i])] for d, i, _, _ in methods]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    w, x = 0.15, np.arange(3)
    for k, ((_, _, lab, c), v) in enumerate(zip(methods, vals)):
        xs = x + (k - 2) * (w + 0.012)
        ax.bar(xs, v, w, color=c, label=lab, zorder=2)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi + 1.5, f"{vi:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Lifetime (FND), % of the oracle")
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7)
    save(fig, "ladder")


def paired():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    xs = np.concatenate([per(BATT, s) for s in "RNBD"]); ys = np.concatenate([per(LEST, s) for s in "RNBD"])
    xi, yi = np.array(intel["Battery Dijkstra"]), np.array(intel[LEST])
    for ax, (x, y, c, title, n) in zip(axes, [(xs, ys, S1, "Synthetic traffic (R, N, B, D)", "120"), (xi, yi, S2, "Intel Lab traffic (1.0 J)", "27")]):
        lo = min(x.min(), y.min()) - 3; hi = 101
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--", zorder=1)
        ax.scatter(x, y, s=16, color=c, edgecolor="white", linewidth=0.6, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("Battery-weighted Dijkstra, % of the oracle")
        ax.set_title(f"{title}, {n} deployments", fontsize=8, color=INK)
        w = int((y > x + 1e-9).sum()); t = int((np.abs(y - x) <= 1e-9).sum()); l = len(x) - w - t
        ax.text(0.04, 0.95, f"above the line: {w}\non the line: {t}\nbelow: {l}", transform=ax.transAxes, va="top", fontsize=7, color=INK)
    axes[0].set_ylabel("LEST load-table Dijkstra, % of the oracle")
    fig.tight_layout()
    save(fig, "paired")


def static_bars():
    rows = [("LP-flow adaptive (LP every 4 rounds)", "LP-flow adaptive (classical)", S1),
            ("RL warm-started from Dijkstra", "RL-Dijkstra warm-start [dijkstra_rl_v2]", S2),
            ("Battery-weighted Dijkstra", "battery Dijkstra (classical)", S1),
            ("LP-flow static (solved once)", "LP-flow static (classical)", S1),
            ("Min-energy Dijkstra", "min-energy Dijkstra", S1),
            ("MADII (rebuild)", "MADII", S2),
            ("FCM clustering", "FCM clustering (paper baseline)", S1)]
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    for k, (lab, key, c) in enumerate(rows):
        v = np.mean(static[key]["pct_ceiling"])
        ax.barh(len(rows) - 1 - k, v, 0.62, color=c, zorder=2)
        ax.text(v + 1, len(rows) - 1 - k, f"{v:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(0, 105); ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_xlabel("Lifetime (FND), % of the LP ceiling")
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=S1), Patch(color=S2)], ["Classical", "Learned"], loc="lower right", fontsize=7)
    save(fig, "static")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ladder(); paired(); static_bars()
    print("figures written to", OUT)
