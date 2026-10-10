"""Results tables for the manuscript, generated from the saved result files so that every number
in the paper matches the data. Run wilcoxon_tables.py first.

Usage: .venv/bin/python paper/make_tables.py   -> paper/body/tab_*.tex
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "body")
J = lambda f: json.load(open(os.path.join(ROOT, f)))

LEST = "Load-aware Dijkstra, LEST load table (4 tiers)"
BATT = "Battery Dijkstra (reactive)"


def p_tex(p, prefix=""):
    if p >= 0.9995:
        body = "1"
    elif p >= 1e-3:
        body = f"{p:.3f}"
    else:
        m, e = f"{p:.1e}".split("e")
        body = f"{m}\\times10^{{{int(e)}}}"
    return f"${prefix}{body}$"


def sgn(v, digits=1):
    """Signed number with a proper minus sign."""
    return f"${v:+.{digits}f}$"


def wtl(x):
    return f"{x['W']}/{x['T']}/{x['L']}"


def write(name, lines):
    open(os.path.join(OUT, name), "w").write("\n".join(lines) + "\n")


def dynamic():
    d = J("results_dynamic.json")
    seeds = sorted(d["R"], key=int)
    per = lambda m, s: np.array([d[s][k][m] for k in seeds])
    rows = [
        ("Proposed", None),
        (r"\textbf{LEST load-table Dijkstra}", LEST),
        ("Load-aware Dijkstra, exact load (no LEST)", "Load-aware Dijkstra (ours, no LP)"),
        ("Upper reference (LP solver every round)", None),
        ("LP planner v2, Holt-Winters", "Forecast LP v2, Holt-Winters (ours)"),
        ("LP planner v2, historical mean (no forecast)", "Forecast LP v2, history mean (no forecast)"),
        ("LP planner v2, perfect forecast (diagnostic)", "Forecast LP v2, perfect forecast (diagnostic)"),
        ("LP planner v1 (every 4 rounds), Holt-Winters", "Forecast LP, Holt-Winters tuned (ours)"),
        ("Classical and published", None),
        ("Battery-weighted Dijkstra", BATT),
        ("Price-guided Dijkstra (daily LP prices)", "Price-guided Dijkstra (daily LP prices)"),
        ("Receiver-weighted Dijkstra", "Battery Dijkstra, receiver-weighted"),
        ("Predictive Dijkstra (Holt-Winters reserve)", "Predictive Dijkstra, Holt-Winters"),
        ("Lifetime Dijkstra (forecast load)", "Lifetime Dijkstra, Holt-Winters (load)"),
        ("Static LP (solved once)", "Static LP (solved once)"),
        ("Reactive LP (last 4 rounds' traffic)", "Reactive LP coordinator"),
        ("EBR-DA~\\cite{mahdi2018ebrda}, as specified", "EBR-DA (Mahdi et al. 2018, as specified)"),
        ("EBR-DA~\\cite{mahdi2018ebrda}, isolated nodes bridged", "EBR-DA (Mahdi et al. 2018, isolated nodes bridged)"),
        ("Learned", None),
        ("MADII~\\cite{yang2025madii} (trained on constant traffic)", "MADII (not retrained)"),
    ]
    out = []
    for lab, m in rows:
        if m is None:
            out.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{lab}}}}}\\\\")
            continue
        v = [per(m, s).mean() for s in "RNBD"]
        allv = np.concatenate([per(m, s) for s in "RNB"])
        x = np.concatenate([per(m, s) for s in "RNBD"]) - np.concatenate([per(BATT, s) for s in "RNBD"])
        w = "--" if m == BATT else f"{(x > 1e-9).sum()}/{(np.abs(x) <= 1e-9).sum()}/{(x < -1e-9).sum()}"
        out.append(f"{lab} & {v[0]:.1f} & {v[1]:.1f} & {v[2]:.1f} & {np.mean(v[:3]):.1f} & {allv.min():.1f} & {v[3]:.1f} & {w}\\\\")
    write("tab_dynamic.tex", out)


def intel():
    a, b = J("results_intel_e1.json")["L"], J("results_intel.json")["L"]
    rows = [(r"\textbf{LEST load-table Dijkstra}", LEST), ("Load-aware Dijkstra, exact load", "Load-aware Dijkstra (ours, no LP)"),
            ("LP planner v2, Holt-Winters", "Forecast LP v2, Holt-Winters (ours)"), ("LP planner v2, historical mean", "Forecast LP v2, history mean (no forecast)"),
            ("LP planner v2, perfect forecast", "Forecast LP v2, perfect forecast (diagnostic)"), ("LP planner v1, Holt-Winters", "Forecast LP v1, Holt-Winters"),
            ("Battery-weighted Dijkstra", "Battery Dijkstra"), ("Reactive LP", "Reactive LP coordinator"), ("Predictive Dijkstra", "Predictive Dijkstra, Holt-Winters"),
            ("Static LP (solved once)", "Static LP (solved once)"), ("EBR-DA, isolated nodes bridged", "EBR-DA (Mahdi et al. 2018, isolated nodes bridged)"),
            ("MADII (not retrained)", "MADII (not retrained)")]
    out = []
    for lab, m in rows:
        cells = []
        for src in (a, b):
            v = np.array(src[m]); r = np.array(src["Battery Dijkstra"]); x = v - r
            w = "--" if m == "Battery Dijkstra" else f"{(x > 1e-9).sum()}/{(np.abs(x) <= 1e-9).sum()}/{(x < -1e-9).sum()}"
            cells += [f"{v.mean():.1f}", f"{v.min():.1f}", w]
        out.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    write("tab_intel.tex", out)


def static():
    c, e = J("results_compare.json"), J("results_eval.json")
    madii = e["madii_v3 (informer, IL)"]
    c["MADII"] = {"pct_ceiling": madii["pct"], "FND": madii["0.01"]["SR"]}
    c["Direct"] = {"pct_ceiling": e["direct"]["pct"], "FND": e["direct"]["0.01"]["SR"]}
    wil = {x["name"]: x for x in J("results_wilcoxon.json")["static_vs_battery"]}
    rows = [("LP-flow adaptive (LP every 4 rounds)", "LP-flow adaptive (classical)", "LP-flow adaptive"),
            ("RL warm-started from Dijkstra", "RL-Dijkstra warm-start [dijkstra_rl_v2]", "RL warm-started from Dijkstra"),
            ("Battery-weighted Dijkstra", "battery Dijkstra (classical)", None),
            ("LP-flow static (solved once)", "LP-flow static (classical)", "LP-flow static"),
            ("Min-energy Dijkstra", "min-energy Dijkstra", "Min-energy Dijkstra"),
            ("MADII rebuild~\\cite{yang2025madii}", "MADII", "MADII"),
            ("Direct to sink", "Direct", None),
            ("FCM clustering (MADII's baseline)", "FCM clustering (paper baseline)", "FCM clustering")]
    out = []
    for lab, k, wk in rows:
        v = np.array(c[k]["pct_ceiling"]); f = np.mean(c[k]["FND"])
        if wk:
            x = wil[wk]; t = f"{wtl(x)}, {p_tex(x['p'], 'p=')}"
        else:
            t = "reference" if "Battery" in lab else "--"
        out.append(f"{lab} & {f:.1f} & {v.mean():.1f} & {v.min():.1f} & {t}\\\\")
    write("tab_static.tex", out)


def wilcoxon():
    w = J("results_wilcoxon.json")
    pick = lambda tab, name: next(x for x in w[tab] if x["name"] == name)
    rows = [("Time-varying traffic, R+N+B+D (120 deployments)", None),
            ("LEST Dijkstra vs battery Dijkstra", pick("dynamic_vs_battery", "LEST load-table Dijkstra")),
            ("LEST Dijkstra vs load-aware Dijkstra, exact load", None, ),
            ("LP planner v2 vs LEST Dijkstra", pick("dynamic_vs_lest", "LP planner v2")),
            ("LP planner v2 vs battery Dijkstra", pick("dynamic_vs_battery", "LP planner v2")),
            ("Price-guided vs battery Dijkstra", pick("dynamic_vs_battery", "Price-guided Dijkstra")),
            ("Predictive vs battery Dijkstra", pick("dynamic_vs_battery", "Predictive Dijkstra")),
            ("Load table only vs LEST Dijkstra", pick("dynamic_vs_lest", "Load table only (no battery)")),
            ("EBR-DA vs LEST Dijkstra", pick("dynamic_vs_lest", "EBR-DA")),
            ("MADII vs LEST Dijkstra", pick("dynamic_vs_lest", "MADII")),
            ("Intel Lab traffic, $E_0=1.0$~J (27 deployments)", None),
            ("LEST Dijkstra vs battery Dijkstra", pick("intel_vs_battery", "LEST load-table Dijkstra")),
            ("LP planner v2 vs LEST Dijkstra", pick("intel_vs_lest", "LP planner v2")),
            ("LP planner v2 vs battery Dijkstra", pick("intel_vs_battery", "LP planner v2")),
            ("Predictive vs battery Dijkstra", pick("intel_vs_battery", "Predictive Dijkstra")),
            ("EBR-DA vs LEST Dijkstra", pick("intel_vs_lest", "EBR-DA")),
            ("MADII vs LEST Dijkstra", pick("intel_vs_lest", "MADII")),
            ("Static traffic, MADII's setting (30 deployments)", None),
            ("RL warm-started vs battery Dijkstra", pick("static_vs_battery", "RL warm-started from Dijkstra")),
            ("LP-flow adaptive vs battery Dijkstra", pick("static_vs_battery", "LP-flow adaptive")),
            ("MADII vs battery Dijkstra", pick("static_vs_battery", "MADII"))]
    exact = pick("dynamic_vs_lest", "Load-aware Dijkstra, exact load")
    out = []
    for r in rows:
        lab, x = r[0], (r[1] if len(r) > 1 else None)
        if lab.startswith("LEST Dijkstra vs load-aware"):          # report LEST minus exact (sign flipped)
            x = dict(exact, mean=-exact["mean"], median=-exact["median"], W=exact["L"], L=exact["W"], r=-exact["r"])
        if x is None:
            out.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{lab}}}}}\\\\")
            continue
        out.append(f"{lab} & {sgn(x['mean'])} & {sgn(x['median'])} & {wtl(x)} & {p_tex(x['p'])} & {p_tex(x['p_holm'])} & {sgn(x['r'], 2)}\\\\")
    write("tab_wilcoxon.tex", out)


if __name__ == "__main__":
    dynamic(); intel(); static(); wilcoxon()
    print("tables written to", OUT)
