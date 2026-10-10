"""Paired Wilcoxon signed-rank tests for the manuscript.

Reads the saved per-deployment results (no simulation is rerun) and compares methods
deployment by deployment:
  - results_compare.json   static traffic, MADII's setting (30 deployments)
  - results_dynamic.json   time-varying traffic, scenarios R, N, B, D (30 deployments each)
  - results_intel_e1.json  Intel Lab traffic, E0 = 1.0 J (27 deployments)

For each pair it reports the median paired difference, wins / ties / losses, the two-sided
Wilcoxon p-value (zero differences dropped, as in scipy's default "wilcox" method), the
matched-pairs rank-biserial correlation r, and the Holm-adjusted p-value within each table.

Usage: .venv/bin/python wilcoxon_tables.py   -> results_wilcoxon.md, results_wilcoxon.json
"""
import json

import numpy as np
from scipy.stats import rankdata, wilcoxon

LEST = "Load-aware Dijkstra, LEST load table (4 tiers)"
BATT = "Battery Dijkstra (reactive)"
LP2 = "Forecast LP v2, Holt-Winters (ours)"

SHORT = {
    LP2: "LP planner v2",
    "Forecast LP v2, history mean (no forecast)": "LP planner v2, historical mean",
    "Forecast LP v2, perfect forecast (diagnostic)": "LP planner v2, perfect forecast",
    "Forecast LP, Holt-Winters tuned (ours)": "LP planner v1",
    LEST: "LEST load-table Dijkstra",
    "Load-aware Dijkstra (ours, no LP)": "Load-aware Dijkstra, exact load",
    BATT: "Battery Dijkstra",
    "Battery Dijkstra": "Battery Dijkstra",
    "Predictive Dijkstra, Holt-Winters": "Predictive Dijkstra",
    "Price-guided Dijkstra (daily LP prices)": "Price-guided Dijkstra",
    "Load table only (ablation, no battery)": "Load table only (no battery)",
    "Static LP (solved once)": "Static LP",
    "Reactive LP coordinator": "Reactive LP",
    "EBR-DA (Mahdi et al. 2018, isolated nodes bridged)": "EBR-DA",
    "MADII (not retrained)": "MADII",
    "Forecast LP v1, Holt-Winters": "LP planner v1",
}


def test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    w, t, l = int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d < -1e-9).sum())
    nz = d[np.abs(d) > 1e-9]
    if len(nz) == 0:
        return dict(n=len(d), median=0.0, mean=0.0, W=w, T=t, L=l, stat=None, p=1.0, r=0.0)
    ranks = rankdata(np.abs(nz))
    r = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    res = wilcoxon(nz)
    return dict(n=len(d), median=float(np.median(d)), mean=float(d.mean()), W=w, T=t, L=l,
                stat=float(res.statistic), p=float(res.pvalue), r=r)


def holm(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m, run = len(rows), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * rows[i]["p"]))
        rows[i]["p_holm"] = run
    return rows


def fmt_p(p):
    return "1" if p >= 0.9995 else (f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}")


def table(title, rows, ref):
    out = [f"### {title}", "", f"Each row: method minus {ref}, paired by deployment.", "",
           "| Method | n | Mean diff | Median diff | W / T / L | Wilcoxon p | Holm p | r |",
           "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        out.append(f"| {x['name']} | {x['n']} | {x['mean']:+.1f} | {x['median']:+.1f} | "
                   f"{x['W']} / {x['T']} / {x['L']} | {fmt_p(x['p'])} | {fmt_p(x['p_holm'])} | {x['r']:+.2f} |")
    return out + [""]


def main():
    dyn = json.load(open("results_dynamic.json"))
    seeds = sorted(dyn["R"], key=int)
    pooled = lambda m, scen: [dyn[s][k][m] for s in scen for k in seeds]
    out, js = ["# Wilcoxon signed-rank tests", ""], {}

    # 1) time-varying traffic, pooled R+N+B+D (120 paired deployments)
    scen = ["R", "N", "B", "D"]
    others = [LP2, "Forecast LP v2, history mean (no forecast)", "Load-aware Dijkstra (ours, no LP)", BATT,
              "Price-guided Dijkstra (daily LP prices)", "Predictive Dijkstra, Holt-Winters",
              "Load table only (ablation, no battery)", "Static LP (solved once)", "Reactive LP coordinator",
              "EBR-DA (Mahdi et al. 2018, isolated nodes bridged)", "MADII (not retrained)"]
    for ref, tag in [(LEST, "lest"), (BATT, "battery")]:
        rows = [dict(name=SHORT[m], **test(pooled(m, scen), pooled(ref, scen))) for m in others if m != ref]
        if ref == BATT:
            rows.insert(1, dict(name=SHORT[LEST], **test(pooled(LEST, scen), pooled(BATT, scen))))
        js[f"dynamic_vs_{tag}"] = holm(rows)
        out += table(f"Time-varying traffic, R+N+B+D pooled (120 deployments), against {SHORT[ref]}", rows, SHORT[ref])

    # 2) per scenario: the three headline comparisons
    rows = []
    for s in scen:
        for a, b in [(LEST, BATT), (LP2, LEST), (LP2, BATT), (LP2, "Forecast LP v2, history mean (no forecast)")]:
            rows.append(dict(name=f"{s}: {SHORT[a]} vs {SHORT[b]}", **test(pooled(a, [s]), pooled(b, [s]))))
    js["per_scenario"] = holm(rows)
    out += table("Per scenario (30 deployments each)", rows, "the second method")

    # 3) Intel Lab, 1.0 J
    it = json.load(open("results_intel_e1.json"))["L"]
    names = [m for m in it if m != "Battery Dijkstra"]
    for ref, tag in [("Load-aware Dijkstra, LEST load table (4 tiers)", "lest"), ("Battery Dijkstra", "battery")]:
        rows = [dict(name=SHORT.get(m, m), **test(it[m], it[ref])) for m in it if m != ref
                and m not in ("Forecast LP v2, seasonal-naive",)]
        js[f"intel_vs_{tag}"] = holm(rows)
        out += table(f"Intel Lab traffic, E0 = 1.0 J (27 deployments), against {SHORT.get(ref, ref)}", rows, SHORT.get(ref, ref))

    # 4) static traffic, MADII's setting
    st = json.load(open("results_compare.json"))
    # the evaluated MADII rebuild (19.5 rounds) is in results_eval.json, on the same 30 deployments
    st["MADII"] = {"pct_ceiling": json.load(open("results_eval.json"))["madii_v3 (informer, IL)"]["pct"]}
    ref = st["battery Dijkstra (classical)"]["pct_ceiling"]
    keep = {"LP-flow adaptive (classical)": "LP-flow adaptive", "RL-Dijkstra warm-start [dijkstra_rl_v2]": "RL warm-started from Dijkstra",
            "LP-flow static (classical)": "LP-flow static", "min-energy Dijkstra": "Min-energy Dijkstra",
            "MADII": "MADII", "FCM clustering (paper baseline)": "FCM clustering"}
    rows = [dict(name=v, **test(st[k]["pct_ceiling"], ref)) for k, v in keep.items()]
    js["static_vs_battery"] = holm(rows)
    out += table("Static traffic, MADII's setting (30 deployments), against battery Dijkstra", rows, "battery Dijkstra")

    out += ["r is the matched-pairs rank-biserial correlation (+1: the method is better on every deployment).",
            "Ties (|difference| < 1e-9) are counted in T and dropped from the test. Holm adjusts within each table."]
    open("results_wilcoxon.md", "w").write("\n".join(out) + "\n")
    json.dump(js, open("results_wilcoxon.json", "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
