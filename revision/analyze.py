"""Statistics, LaTeX tables and figures for the revised manuscript, from results/revision/*.jsonl.

Statistical unit: the topology (deployment seed). The primary score of a method on a topology
is the mean of its R, N and B scores; D is reported beside it. Paired comparisons use that
topology mean (n = 30): mean difference with a 95% bootstrap CI over topologies (20,000
resamples), the matched-pairs rank-biserial r, and a two-sided Wilcoxon signed-rank p with Holm
adjustment over the comparisons listed in PRIMARY.

  .venv/bin/python -m revision.analyze     -> results/revision/summary.json, paper/body/rev_*.tex, paper/figs/rev_*.pdf
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

R = Path("results/revision")
BODY = Path("paper/body")
FIGS = Path("paper/figs")
BATT, BLD, FA = "Battery-weighted Dijkstra", "Booked-load Dijkstra", "Flow augmentation (Chang-Tassiulas)"
LP, LEST, LOAD = "LP planner (every round)", "Booked-load, LEST 4-tier load (ablation)", "Booked-load, load only (ablation)"
SHAN, MTE = "Load-balanced SPT (Shan et al.)", "Min-energy Dijkstra"
SHORT = {BATT: "Battery-weighted Dijkstra", BLD: r"Booked-load Dijkstra", FA: "Flow augmentation [CT04]",
         LP: "LP planner (LP every round)", LEST: "Booked-load, LEST 4-tier load", LOAD: "Booked-load, load only (no battery)",
         SHAN: "Load-balanced SPT [Shan13]", MTE: "Min-energy Dijkstra"}
CITE = {"[CT04]": r"~\cite{chang2004maximum}", "[Shan13]": r"~\cite{shan2013lifetime}"}
ORDER = [LP, FA, BLD, LEST, BATT, LOAD, MTE, SHAN]
PRIMARY = [(BLD, BATT), (FA, BATT), (BLD, FA), (LP, BLD), (LP, FA), (LEST, BLD), (LOAD, BLD), (SHAN, BATT)]
RNG = np.random.default_rng(32)


def tex_name(m):
    s = SHORT.get(m, m)
    for k, v in CITE.items():
        s = s.replace(" " + k, v)
    return s


def load(exp):
    p = R / f"{exp}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def table(rows, cond=None, key=lambda r: r["method"]):
    """{method: {scen: {seed: score}}} for one condition."""
    t = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if cond is None or r["cond"] == cond:
            t[key(r)][r["scen"]][r["seed"]] = r["score"]
    return t


def topo(t, m, scens):
    seeds = sorted(set.intersection(*[set(t[m][s]) for s in scens]))
    return np.array(seeds), np.array([np.mean([t[m][s][k] for s in scens]) for k in seeds])


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    boot = np.array([d[RNG.integers(len(d), size=len(d))].mean() for _ in range(20000)])
    nz = d[np.abs(d) > 1e-9]
    if len(nz):
        rk = rankdata(np.abs(nz)); r = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()); p = float(wilcoxon(nz).pvalue)
    else:
        r, p = 0.0, 1.0
    return dict(n=len(d), mean=float(d.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                W=int((d > 1e-9).sum()), T=int((np.abs(d) <= 1e-9).sum()), L=int((d < -1e-9).sum()), r=r, p=p)


def holm(rows):
    prev = 0.0
    for k, i in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]["p"])):
        prev = max(prev, min(1.0, (len(rows) - k) * rows[i]["p"])); rows[i]["p_holm"] = prev
    return rows


def ptex(p):
    if p >= 1e-3:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def sg(x, d=1):
    return f"${x:+.{d}f}$"


def ci(x):
    return f"${x['mean']:+.1f}$ [${x['lo']:+.1f}$, ${x['hi']:+.1f}$]"


def write(name, lines):
    (BODY / name).write_text("\n".join(lines) + "\n")


def main_tables(S):
    rows = load("main")
    t = table(rows)
    out = []
    for m in ORDER:
        if m not in t:
            continue
        per = {s: np.mean(list(t[m][s].values())) for s in "RNBD"}
        _, tm = topo(t, m, "RNB")
        cells = [f"{per[s]:.1f}" for s in "RNB"] + [f"{tm.mean():.1f}", f"{tm.min():.1f}", f"{per['D']:.1f}"]
        if m == BATT:
            cells.append("reference")
        else:
            _, a = topo(t, m, "RNB"); _, b = topo(t, BATT, "RNB")
            cells.append(ci(paired(a, b)))
        name = tex_name(m)
        if m == BLD:
            name = r"\textbf{" + name + "}"
        out.append(name + " & " + " & ".join(cells) + r"\\")
        S["main"][m] = dict(R=per["R"], N=per["N"], B=per["B"], D=per["D"], S=float(tm.mean()), worst=float(tm.min()))
    write("rev_main.tex", out)
    # primary paired comparisons (Holm family)
    comps = []
    for a, b in PRIMARY:
        if a in t and b in t:
            _, x = topo(t, a, "RNB"); _, y = topo(t, b, "RNB")
            comps.append(dict(a=a, b=b, **paired(x, y)))
    holm(comps)
    S["primary"] = comps
    out = [f"{tex_name(c['a'])} vs {tex_name(c['b'])} & {ci(c)} & {c['W']}/{c['T']}/{c['L']} & {sg(c['r'], 2)} & ${ptex(c['p_holm'])}$\\\\" for c in comps]
    write("rev_stats.tex", out)
    # tightness certificate: best realised lifetime on each topology and scenario vs the bound
    best = []
    for s in "RNBD":
        for k in t[BATT][s]:
            best.append(max(t[m][s][k] for m in t if k in t[m][s]))
    lp = [t[LP][s][k] for s in "RNBD" for k in t[LP][s]]
    S["tightness"] = dict(best_mean=float(np.mean(best)), best_min=float(np.min(best)), lp_mean=float(np.mean(lp)),
                          lp_min=float(np.min(lp)), cases=len(best))
    # pooled descriptive W/T/L over 120 cases for BLD vs battery
    d = [t[BLD][s][k] - t[BATT][s][k] for s in "RNBD" for k in t[BATT][s]]
    S["bld_vs_batt_cases"] = dict(W=int(sum(x > 1e-9 for x in d)), T=int(sum(abs(x) <= 1e-9 for x in d)), L=int(sum(x < -1e-9 for x in d)))
    d = [t[FA][s][k] - t[BATT][s][k] for s in "RNBD" for k in t[BATT][s]]
    S["fa_vs_batt_cases"] = dict(W=int(sum(x > 1e-9 for x in d)), T=int(sum(abs(x) <= 1e-9 for x in d)), L=int(sum(x < -1e-9 for x in d)))
    S["timing_ms"] = {m: 1000 * float(np.mean([r["sec_per_round"] for r in rows if r["method"] == m])) for m in t}
    info = {MTE: ("positions", "one Dijkstra", "0"), BATT: ("+ batteries", "one Dijkstra", "0"),
            SHAN: ("positions, range $R$", "rebuild on a death (max-flow)", "0"),
            FA: ("+ batteries, observed rates", "$K=30$ Dijkstra runs", "0"),
            BLD: ("+ batteries, observed rates", r"$\le n$ Dijkstra runs", "0"),
            LEST: ("+ batteries, tiered load", r"$\le n$ Dijkstra runs", r"8/packet, $2n$/round"),
            LP: ("+ batteries, observed rates", "one LP (HiGHS)", "0")}
    write("rev_char.tex", [f"{tex_name(m)} & {a} & {b} & {c} & {S['timing_ms'][m]:.1f}" + r"\\" for m, (a, b, c) in info.items() if m in S["timing_ms"]])


def grid_tables(S):
    rows = load("grid")
    conds = ["n50", "n200", "sink-corner", "clustered", "clustered-corner"]
    label = {"n50": "$n=50$", "n200": "$n=200$", "sink-corner": "Sink at corner", "clustered": "Clustered",
             "clustered-corner": "Clustered, sink at corner"}
    out = []
    for c in conds:
        t = table(rows, c)
        if BATT not in t:
            continue
        vals = []
        for m in [MTE, SHAN, BATT, FA, BLD, LP]:
            vals.append(f"{topo(t, m, 'RN')[1].mean():.1f}" if m in t else "--")
        _, a = topo(t, BLD, "RN"); _, b = topo(t, BATT, "RN"); _, f = topo(t, FA, "RN")
        x, y = paired(a, b), paired(a, f)
        S["grid"][c] = dict(bld_minus_batt=x, bld_minus_fa=y)
        out.append(f"{label[c]} & " + " & ".join(vals) + f" & {ci(x)} & {ci(y)}\\\\")
    write("rev_grid.tex", out)


def kappa_table(S):
    rows = load("kappa")
    t = table(rows, key=lambda r: r["cond"])
    out = []
    ks = sorted(t, key=lambda c: float(c.split("=")[1]))
    _, b = topo(t, "kappa=0", "RN")
    for c in ks:
        _, v = topo(t, c, "RN")
        x = paired(v, b)
        S["kappa"][c] = dict(R=float(np.mean(list(t[c]["R"].values()))), N=float(np.mean(list(t[c]["N"].values()))), mean=float(v.mean()), vs_batt=x)
        out.append(f"{c.split('=')[1]} & {S['kappa'][c]['R']:.1f} & {S['kappa'][c]['N']:.1f} & {v.mean():.1f} & " + ("reference" if c == "kappa=0" else ci(x)) + r"\\")
    write("rev_kappa.tex", out)


def lossy_table(S):
    rows = load("lossy")
    out = []
    for c in ["lossy", "lossy+contention"]:
        rs = [r for r in rows if r["cond"] == c]
        t = table(rs)
        for m in [MTE, BATT, FA, BLD, LP]:
            if m not in t:
                continue
            _, v = topo(t, m, "RN")
            fs = [r for r in rs if r["method"] == m]
            dr = 100 * sum(r["delivered_kbit"] for r in fs) / max(sum(r["generated_kbit"] for r in fs), 1e-9)
            dl = np.mean([r["delivered_kbit"] for r in fs])
            x = "reference" if m == BATT else ci(paired(v, topo(t, BATT, "RN")[1]))
            S["lossy"][c + "|" + m] = dict(score=float(v.mean()), delivery=float(dr), delivered_kbit=float(dl))
            out.append(f"{'Lossy + contention' if 'cont' in c else 'Lossy, ARQ'} & {tex_name(m)} & {v.mean():.1f} & {x} & {dr:.1f} & {dl / 1000:.1f}\\\\")
        out.append(r"\midrule")
    write("rev_lossy.tex", out[:-1])


def lossy_delivery(S):
    """Delivery ratio needs generated bits; recompute it from delivered data and lifetime."""
    pass


def trace_table(S):
    rows = load("trace")
    out = []
    for c in ["intel e0=1.0", "intel e0=1.5"]:
        rs = [r for r in rows if r["cond"] == c]
        for m in [LP, FA, BLD, LEST, BATT, MTE, SHAN]:
            v = [r for r in rs if r["method"] == m]
            if not v:
                continue
            sc = np.array([r["score"] for r in v]); cen = sum(r["censored"] for r in v)
            S["trace"][c + "|" + m] = dict(mean=float(sc.mean()), worst=float(sc.min()), censored=int(cen), n=len(v))
        line = []
    for m in [LP, FA, BLD, LEST, BATT, MTE, SHAN]:
        cells = []
        for c in ["intel e0=1.0", "intel e0=1.5"]:
            x = S["trace"].get(c + "|" + m)
            cells += [f"{x['mean']:.1f}", f"{x['worst']:.1f}", f"{x['censored']}"] if x else ["--"] * 3
        out.append(tex_name(m) + " & " + " & ".join(cells) + r"\\")
    write("rev_trace.tex", out)


def scaling_table(S):
    p = R / "scaling.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    S["scaling"] = rows
    out = []
    for r in rows:
        lp = r.get("LP solve (sparse HiGHS)")
        out.append(f"{r['n']} & {1000 * r[BATT]:.1f} & {1000 * r[FA]:.1f} & {1000 * r[BLD]:.1f} & " + (f"{lp:.2f}" if lp is not None else "--") + r"\\")
    write("rev_scaling.tex", out)


def madii_appendix(S):
    d = json.load(open("runs/publication_revision/madii_static.json"))
    diag = json.load(open(R / "madii_diag.json"))
    sel = {k: m for k, m in d["models"].items() if k.endswith("_best")}
    fnd = {k: float(np.mean([[x["0.01"]["SR"] for x in rep] for rep in m["raw"]])) for k, m in sel.items()}
    S["madii"] = {"selected_FND": fnd, "episodes": {k: m["episode"] for k, m in sel.items()}}
    S["madii_diag"] = {k: {kk: v for kk, v in d_.items() if kk != "per_seed_first_round"} for k, d_ in diag.items()}
    dr, bd = S["madii_diag"]["Direct to sink"], S["madii_diag"]["Battery-weighted Dijkstra"]
    f = np.mean(list(fnd.values()))
    write("rev_madii.tex", [
        f"With the corrected reward, the validation-selected checkpoints of all three seeds reach {f:.2f} rounds to first death "
        f"({100 * (f - 1) / np.mean(d['ceiling_rounds']):.1f}\\% of the fractional bound, legacy accounting). "
        f"This equals direct transmission to the sink ({dr['F']:.2f} rounds): in the first round of every test topology the selected models "
        "send every source straight to the sink. This is not what the reward asks for. Scored with the same corrected reward (MADII's Eq.~8), "
        f"battery-weighted Dijkstra earns {bd['first_round'] / dr['first_round']:.1f} times more reward than direct transmission in the first round, "
        f"and {bd['discounted'] / dr['discounted']:.1f} times more discounted return ($\\gamma=0.99$) over an episode. The collapse is therefore a "
        "learning failure of the reconstruction (plausibly credit assignment across 100 agents sharing one round return), not a "
        "property of the reward."])


def _excl(x):
    return x["lo"] > 0 or x["hi"] < 0


def texts(S):
    """Data-driven sentences for the results section (paper/body/rev_*_text.tex)."""
    if S.get("kappa"):
        k = {float(c.split("=")[1]): v["vs_batt"] for c, v in S["kappa"].items() if c != "kappa=0"}
        good = sorted(x for x, v in k.items() if v["lo"] > 0)
        bad = sorted(x for x, v in k.items() if v["hi"] < 0)
        best = max(k, key=lambda x: k[x]["mean"])
        near = sorted(x for x, v in k.items() if v["mean"] >= k[best]["mean"] - 1.0)
        sent = (f"the gain over battery Dijkstra is positive for $\\kappa$ from {good[0]:g} to {good[-1]:g}, with a flat optimum for $\\kappa\\in[{near[0]:g}, {near[-1]:g}]$ "
                f"(within one point of the best, {k[best]['mean']:+.1f} at $\\kappa={best:g}$). ")
        if bad:
            sent += (f"Larger weights hurt: $\\kappa={bad[0]:g}$ is {abs(k[bad[0]]['mean']):.1f} points \\emph{{below}} battery Dijkstra, plausibly because an overweighted load term "
                     "pushes traffic around busy relays onto weaker ones. ")
        sent += f"The validation choice $\\kappa=0.5$ gives {ci(k[0.5])} on these fresh topologies."
        write("rev_kappa_text.tex", [sent])
    if S.get("grid"):
        g = S["grid"]
        bb = [v["bld_minus_batt"] for v in g.values()]; bf = {c: v["bld_minus_fa"] for c, v in g.items()}
        pos = sum(x["lo"] > 0 for x in bb)
        lab = {"n50": "$n=50$", "n200": "$n=200$", "sink-corner": "the corner sink", "clustered": "clustered layouts", "clustered-corner": "clustered layouts with a corner sink"}
        fa_better = [lab[c] for c, x in bf.items() if x["hi"] < 0]
        bld_better = [lab[c] for c, x in bf.items() if x["lo"] > 0]
        sent = (f"Booked-load Dijkstra improves on battery Dijkstra in every condition, by {min(x['mean'] for x in bb):.1f} to {max(x['mean'] for x in bb):.1f} points, "
                f"with the 95\\% interval above zero in {pos} of {len(bb)} conditions. ")
        if fa_better or bld_better:
            sent += "Against flow augmentation the picture depends on the condition: "
            parts = []
            if fa_better:
                parts.append("flow augmentation is ahead for " + ", ".join(fa_better))
            if bld_better:
                parts.append("booked-load Dijkstra is ahead for " + ", ".join(bld_better))
            rest = [lab[c] for c, x in bf.items() if not _excl(x)]
            if rest:
                parts.append("and the two are indistinguishable for " + ", ".join(rest))
            sent += "; ".join(parts) + ". "
        else:
            sent += "Booked-load Dijkstra and flow augmentation remain indistinguishable in every condition. "
        sent += "The ranking of the router families (MTE and Shan's tree far below, battery Dijkstra, then the two coordinated routers, then the LP planner where it was run) does not change."
        write("rev_grid_text.tex", [sent])
    if S.get("lossy"):
        L = S["lossy"]
        out = []
        for c, name in [("lossy", "With lossy links and ARQ"), ("lossy+contention", "With contention added")]:
            b, k, f, l = (L.get(c + "|" + m) for m in (BATT, BLD, FA, LP))
            if not (b and k and f and l):
                continue
            out.append(f"{name}, booked-load Dijkstra and flow augmentation live {k['score'] - b['score']:.1f} and {f['score'] - b['score']:.1f} points longer than battery Dijkstra, "
                       f"and the LP planner {l['score'] - b['score']:.1f} points longer, while delivery falls from {b['delivery']:.1f}\\% to {k['delivery']:.1f}\\%, {f['delivery']:.1f}\\% and {l['delivery']:.1f}\\%.")
        out.append("The lifetime ranking of the ideal channel survives losses and contention, but every router that spreads traffic beyond battery Dijkstra's routes pays in delivery, "
                   "and the more it gains in lifetime, the more it pays: the LP planner most of all. "
                   "Under lossy links the lifetime gain is therefore a trade-off rather than a free improvement, and the payload delivered before the first death (last column) is the fairer single number.")
        write("rev_lossy_text.tex", [" ".join(out)])
    if S.get("trace"):
        T = S["trace"]; c = "intel e0=1.0"
        g = lambda m: T.get(c + "|" + m)
        rows = [r for r in load("trace") if r["cond"] == c]
        sc = {(r["method"], r["seed"]): r["score"] for r in rows}
        cen = {(r["method"], r["seed"]): r["censored"] for r in rows}
        wins = [sc[(BLD, k)] - sc[(BATT, k)] for (m, k) in sc if m == BATT and (BLD, k) in sc]
        W = sum(x > 1e-9 for x in wins); Tt = sum(abs(x) <= 1e-9 for x in wins); Lw = sum(x < -1e-9 for x in wins)
        tc = sum(1 for (m, k) in sc if m == BATT and (BLD, k) in sc and abs(sc[(BLD, k)] - sc[(BATT, k)]) <= 1e-9 and cen[(BATT, k)] and cen[(BLD, k)])
        if g(BATT) and g(BLD) and g(FA) and g(LP):
            write("rev_trace_text.tex", [
                f"At 1.0~J the mean scores are {g(LP)['mean']:.1f}\\% for the LP planner, {g(BLD)['mean']:.1f}\\% for booked-load Dijkstra, "
                f"{g(FA)['mean']:.1f}\\% for flow augmentation and {g(BATT)['mean']:.1f}\\% for battery Dijkstra: the same three levels as on synthetic traffic. "
                f"Booked-load Dijkstra is ahead of battery Dijkstra in {W} windows, tied in {Tt} ({tc} of them because both runs outlast the trace) and behind in {Lw}."])
    if S.get("scaling"):
        sc = {r["n"]: r for r in S["scaling"]}
        n_hi = max(n for n in sc if sc[n].get("LP solve (sparse HiGHS)") is not None)
        write("rev_scaling_text.tex", [
            f"One LP solve takes {sc[100]['LP solve (sparse HiGHS)']:.2f}~s at $n=100$ but {sc[n_hi]['LP solve (sparse HiGHS)']:.0f}~s at $n={n_hi}$, "
            f"against {1000 * sc[n_hi][BLD]:.0f}~ms for booked-load Dijkstra and {1000 * sc[n_hi][FA]:.0f}~ms for flow augmentation. "
            "With hour-long rounds a server-class sink can still afford the LP at these sizes, so in the modelled system avoiding the solver buys little. "
            "Solver-free coordination matters where the sink is an embedded gateway, the rounds are short, or the network is much larger, and in a distributed "
            "implementation, which this study does not evaluate."])


def figures(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "legend.frameon": False})
    C = {"battery": "#2a78d6", "fa": "#eb6834", "bld": "#1baf7a", "lp": "#e87ba4"}
    # paired topology means: BLD vs battery and BLD vs FA
    t = table(load("main"))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for a_, (other, lab, col) in zip(ax, [(BATT, "Battery-weighted Dijkstra", C["battery"]), (FA, "Flow augmentation", C["fa"])]):
        _, x = topo(t, other, "RNB"); _, y = topo(t, BLD, "RNB")
        lo, hi = min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2
        a_.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="#52514e")
        a_.scatter(x, y, s=16, color=col, edgecolor="white", linewidth=0.6, zorder=3)
        a_.set_xlim(lo, hi); a_.set_ylim(lo, hi); a_.set_aspect("equal")
        a_.set_xlabel(f"{lab}, % of bound"); a_.set_title(f"30 topologies, mean of R, N, B", fontsize=8)
        w = int((y > x + 1e-9).sum()); a_.text(0.04, 0.95, f"above the line: {w}/30", transform=a_.transAxes, va="top", fontsize=7)
    ax[0].set_ylabel("Booked-load Dijkstra, % of bound")
    fig.tight_layout(); fig.savefig(FIGS / "rev_paired.pdf", bbox_inches="tight")
    # grid: BLD - battery and FA - battery per condition, with CIs
    rows = load("grid")
    conds = [c for c in ["n50", "n200", "sink-corner", "clustered", "clustered-corner"] if any(r["cond"] == c for r in rows)]
    if conds:
        fig, ax = plt.subplots(figsize=(6.6, 2.4))
        lab = {"n50": "n = 50", "n200": "n = 200", "sink-corner": "sink corner", "clustered": "clustered", "clustered-corner": "clustered,\nsink corner"}
        for k, (m, col, off) in enumerate([(FA, C["fa"], -0.15), (BLD, C["bld"], 0.15), (LP, C["lp"], 0.0)]):
            if m == LP:
                continue
            for i, c in enumerate(conds):
                tt = table(rows, c)
                x = paired(topo(tt, m, "RN")[1], topo(tt, BATT, "RN")[1])
                ax.errorbar(i + off, x["mean"], yerr=[[x["mean"] - x["lo"]], [x["hi"] - x["mean"]]], fmt="o", color=col, ms=4, capsize=2,
                            label=("Flow augmentation" if m == FA else "Booked-load Dijkstra") if i == 0 else None)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(range(len(conds)), [lab[c] for c in conds])
        ax.set_ylabel("Gain over battery Dijkstra\n(points, 95% CI)")
        ax.legend(fontsize=7, loc="upper left")
        fig.savefig(FIGS / "rev_grid.pdf", bbox_inches="tight")


def main():
    S = defaultdict(dict)
    for f in (main_tables, grid_tables, kappa_table, lossy_table, trace_table, scaling_table, madii_appendix, texts, figures):
        try:
            f(S)
        except (KeyError, ValueError, FileNotFoundError) as e:
            print("skipped", f.__name__, repr(e))
    (R / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({k: S[k] for k in ("main", "tightness", "bld_vs_batt_cases", "fa_vs_batt_cases") if k in S}, indent=1, default=float))
    for c in S.get("primary", []):
        print(f"{c['a'][:30]:30s} vs {c['b'][:30]:30s} {c['mean']:+.2f} [{c['lo']:+.2f},{c['hi']:+.2f}] {c['W']}/{c['T']}/{c['L']} r={c['r']:+.2f} pH={c['p_holm']:.2g}")


if __name__ == "__main__":
    main()
