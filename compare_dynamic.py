"""Part 3 evaluation: every router on time-varying traffic, scored against the oracle LP.

Scenarios (traffic.py): R = daily hotspot surge, N = constant traffic, B = surge + random
bursts. 30 held-out deployments (seeds 0-29) per scenario; traffic seed = deployment + 1000.
Score L(m, s) = (rounds to first node death - 1) / oracle T for that deployment and traffic.

Fixed a priori, not tuned on the evaluation seeds: forecast horizon H = 24 rounds (one day),
LP re-solved every 4 rounds; predictive Dijkstra uses lam = 0.5, H = 12, the best non-zero
setting on validation deployments 500-505.

  .venv/bin/python compare_dynamic.py --workers 4     # writes results_dynamic.{json,md}
"""
import argparse, json
from multiprocessing import Pool

import numpy as np
from scipy.stats import wilcoxon

import env as E
import dijkstra_rl as D
import traffic as TR
from lp_oracle import oracle_T
from predictive import ForecastLP, PredictiveDijkstra, ReactiveLP, StaticLP

SCEN = {"R": "Forecast right (daily surge)", "N": "Not needed (constant)", "B": "Sudden burst"}
METHODS = ["Static LP (solved once)", "Reactive LP coordinator", "Battery Dijkstra (reactive)", "MADII (not retrained)",
           "Predictive Dijkstra, Holt-Winters", "Forecast LP, persistence", "Forecast LP, seasonal-naive",
           "Forecast LP, Holt-Winters (ours)", "Forecast LP, perfect forecast (diagnostic)"]
_net = None


def _madii():
    global _net
    if _net is None:
        import torch
        torch.set_num_threads(1); torch.manual_seed(0)
        from evaluate import load
        _net = load("checkpoints/madii_v3_best.pt")[0]
    from train import greedy_action
    return lambda w: greedy_action(_net, w)


def build(name, w, h, tr):
    return {"Static LP (solved once)": lambda: StaticLP(w, h),
            "Reactive LP coordinator": lambda: ReactiveLP(w, h),
            "Battery Dijkstra (reactive)": lambda: D.fast_battery_weighted,
            "MADII (not retrained)": _madii,
            "Predictive Dijkstra, Holt-Winters": lambda: PredictiveDijkstra(w, h, lam=0.5, H=12),
            "Forecast LP, persistence": lambda: ForecastLP(w, h, forecaster="persistence"),
            "Forecast LP, seasonal-naive": lambda: ForecastLP(w, h, forecaster="seasonal-naive"),
            "Forecast LP, Holt-Winters (ours)": lambda: ForecastLP(w, h),
            "Forecast LP, perfect forecast (diagnostic)": lambda: ForecastLP(w, h, forecaster="perfect", future=tr)}[name]()


def job(args):
    scen, s = args
    w0 = E.WSN(seed=s)
    h, tr = TR.generate(scen, w0.pos[:w0.n], s + 1000)
    To = oracle_T(E.WSN(seed=s), tr)
    out = {"oracle": To}
    for m in METHODS:
        w = E.WSN(seed=s, traffic=tr)
        pol = build(m, w, h, tr)
        while w.round < len(tr):
            if w.step(pol(w))["n_dead"] >= 1:
                break
        out[m] = (w.round - 1) / To * 100
    return scen, s, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summarize-only", action="store_true", help="rebuild results_dynamic.md from the JSON")
    a = ap.parse_args()
    if a.summarize_only:
        raw = json.load(open("results_dynamic.json"))
        summarize({sc: {int(s): v for s, v in raw[sc].items()} for sc in raw}, a.seeds)
        return
    jobs = [(sc, s) for sc in SCEN for s in range(a.seeds)]
    res = {sc: {} for sc in SCEN}
    with Pool(a.workers) as p:
        for sc, s, out in p.imap_unordered(job, jobs):
            res[sc][s] = out
            print(sc, s, {k: round(v, 1) for k, v in out.items()}, flush=True)
    json.dump({sc: {str(s): res[sc][s] for s in res[sc]} for sc in res}, open("results_dynamic.json", "w"), indent=1)
    summarize(res, a.seeds)


def break_even(G, CN, CB):
    """p* from G (gain when the forecast is right) and the losses C(N), C(B) elsewhere."""
    Cb = (CN + CB) / 2
    if G <= 0:
        return "no gain when the forecast is right, so prediction does not pay off as a forecast"
    if CN <= 0 and CB <= 0:
        return "p* = 0: it wins in every scenario, so it pays at any forecast accuracy"
    if Cb <= 0:
        return "p* = 0 on equal weights: it loses in one scenario, but its gains elsewhere outweigh that loss"
    return f"p* = {Cb / (G + Cb):.2f}: it pays when the forecast is right more than {100 * Cb / (G + Cb):.0f}% of the time"


def summarize(res, seeds):
    methods = [m for m in METHODS if all(m in res[sc][0] for sc in SCEN)] + \
              [m for m in res["R"][0] if m not in METHODS and m != "oracle"]
    L = {m: {sc: np.array([res[sc][s][m] for s in range(seeds)]) for sc in SCEN} for m in methods}
    ref = "Battery Dijkstra (reactive)"
    lines = [f"{seeds} held-out deployments per scenario; L = lifetime to first node death, % of the oracle LP.", "",
             "| Method | " + " | ".join(SCEN.values()) + " | Mean S (equal weights) | Worst % |", "|---|" + "---|" * (len(SCEN) + 2)]
    for m in methods:
        v = [L[m][sc].mean() for sc in SCEN]
        lines.append(f"| {m} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {np.mean(v):.1f} | {min(L[m][sc].min() for sc in SCEN):.1f} |")
    lines += ["", f"Paired vs {ref} (wins/ties/losses, Wilcoxon p):", ""]
    for m in methods:
        if m == ref:
            continue
        cells = []
        for sc in SCEN:
            d = L[m][sc] - L[ref][sc]
            p = wilcoxon(L[m][sc], L[ref][sc]).pvalue if np.any(d != 0) else 1.0
            cells.append(f"{sc}: {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}, p={p:.2g}")
        lines.append(f"- {m}: " + "; ".join(cells))
    lines += ["", "Break-even forecast accuracy (G = gain when the forecast is right; C = loss when not needed / burst):", ""]
    for m in [m for m in methods if m.startswith("Forecast LP") or m.startswith("Predictive")]:
        for base in (ref, "Reactive LP coordinator"):
            G = L[m]["R"].mean() - L[base]["R"].mean()
            CN = L[base]["N"].mean() - L[m]["N"].mean()
            CB = L[base]["B"].mean() - L[m]["B"].mean()
            lines.append(f"- {m} vs {base}: G = {G:+.1f}, C(N) = {CN:+.1f}, C(B) = {CB:+.1f} -> {break_even(G, CN, CB)}")
    md = "\n".join(lines)
    print("\n" + md)
    open("results_dynamic.md", "w").write(md + "\n")


if __name__ == "__main__":
    main()
