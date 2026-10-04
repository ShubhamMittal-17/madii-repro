"""Part 3 evaluation: every router on time-varying traffic, scored against the oracle LP.

Scenarios (traffic.py): R = daily hotspot surge, N = constant traffic, B = surge + random
bursts. 30 held-out deployments (seeds 0-29) per scenario; traffic seed = deployment + 1000.
Score L(m, s) = (rounds to first node death - 1) / oracle T for that deployment and traffic.

Fixed a priori, not tuned on the evaluation seeds: forecast horizon H = 24 rounds (one day),
LP re-solved every 4 rounds. Tuned on validation deployments 500-505 only: predictive Dijkstra
lam = 0.5, H = 12 (best non-zero setting), lifetime Dijkstra load k = 0.5 and tau rho = 0.1,
k = 0.25 (best non-zero settings; every larger k was worse, and k = 0 is battery Dijkstra), and Holt-Winters alpha = 0.05, beta = 0, gamma = 0.1
(grid alpha in {0.02, 0.05, 0.1, 0.2} x gamma in {0.1, 0.3, 0.6}). The a-priori Holt-Winters
setting (alpha 0.2, beta 0.01, gamma 0.3) is kept as a row for transparency.

Forecast LP v2 (tune_forecast_lp.py, validation deployments 500-511 only): the same forecast
LP with the planner fixes that held up on validation (re-solved every round, as battery
Dijkstra re-plans every round, and a min-energy tie-break among max-lifetime flows). Its
"history mean" row is the ablation that feeds the identical planner a fixed historical rate
instead of a forecast, which isolates what forecasting adds.

Robustness scenario D (pattern shift; nothing is tuned on it): the hotspot moves to a new
spot two days into the deployment. It is reported beside the score, not inside S, so the
break-even formula keeps its R / N / B definition.

  .venv/bin/python compare_dynamic.py --workers 4     # writes results_dynamic.{json,md}
  .venv/bin/python compare_dynamic.py --scen D        # adds the robustness scenario
"""
import argparse, json
from multiprocessing import Pool

import numpy as np
from scipy.stats import wilcoxon

import env as E
import dijkstra_rl as D
import traffic as TR
from lp_oracle import oracle_T
from predictive import EBRDA, ForecastLP, LifetimeDijkstra, LESTDijkstra, LoadAwareDijkstra, PredictiveDijkstra, PriceDijkstra, ReactiveLP, StaticLP

SCEN = {"R": "Forecast right (daily surge)", "N": "Not needed (constant)", "B": "Sudden burst"}
EXTRA_SCEN = {"D": "Pattern shift (robustness)"}
METHODS = ["Static LP (solved once)", "Reactive LP coordinator", "Battery Dijkstra (reactive)", "MADII (not retrained)",
           "Predictive Dijkstra, Holt-Winters", "Lifetime Dijkstra, Holt-Winters (load)", "Lifetime Dijkstra, Holt-Winters (tau)",
           "Forecast LP, persistence", "Forecast LP, seasonal-naive",
           "Forecast LP, Holt-Winters (a priori)", "Forecast LP, Holt-Winters tuned (ours)", "Forecast LP, perfect forecast (diagnostic)",
           "Forecast LP v2, Holt-Winters (ours)", "Forecast LP v2, history mean (no forecast)", "Forecast LP v2, perfect forecast (diagnostic)",
           "Battery Dijkstra, receiver-weighted", "Price-guided Dijkstra (daily LP prices)", "Load-aware Dijkstra (ours, no LP)",
           "Load table only (ablation, no battery)", "Load-aware Dijkstra, LEST load table (4 tiers)",
           "EBR-DA-style (energy + load cost, reconstructed)"]
HW_TUNED = dict(alpha=0.05, beta=0.0, gamma=0.1)   # fitted on validation deployments 500-505, R and B
V2 = dict(resolve_every=1, mu=1e-3)                # planner fixes chosen on validation 500-511
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
            "Lifetime Dijkstra, Holt-Winters (load)": lambda: LifetimeDijkstra(w, h, mode="load", k=0.5, **HW_TUNED),
            "Lifetime Dijkstra, Holt-Winters (tau)": lambda: LifetimeDijkstra(w, h, mode="tau", rho=0.1, k=0.25, **HW_TUNED),
            "Forecast LP, persistence": lambda: ForecastLP(w, h, forecaster="persistence"),
            "Forecast LP, seasonal-naive": lambda: ForecastLP(w, h, forecaster="seasonal-naive"),
            "Forecast LP, Holt-Winters (a priori)": lambda: ForecastLP(w, h),
            "Forecast LP, Holt-Winters tuned (ours)": lambda: ForecastLP(w, h, **HW_TUNED),
            "Forecast LP, perfect forecast (diagnostic)": lambda: ForecastLP(w, h, forecaster="perfect", future=tr),
            "Battery Dijkstra, receiver-weighted": lambda: PriceDijkstra(w, h, c=0, rx="receiver", **HW_TUNED),
            "Load-aware Dijkstra (ours, no LP)": lambda: LoadAwareDijkstra(w, h, kappa=1.0, order="near"),
            "Load table only (ablation, no battery)": lambda: LoadAwareDijkstra(w, h, kappa=1.0, order="near", energy=False),
            "Load-aware Dijkstra, LEST load table (4 tiers)": lambda: LESTDijkstra(w, h, levels="lest", rule="original", band=0.05, exact_energy=True),
            "EBR-DA-style (energy + load cost, reconstructed)": lambda: EBRDA(w, h, a=1.0, c=1000.0),
            "Price-guided Dijkstra (daily LP prices)": lambda: PriceDijkstra(w, h, c=0.3, every=24, rx="receiver", **HW_TUNED),
            "Forecast LP v2, Holt-Winters (ours)": lambda: ForecastLP(w, h, **HW_TUNED, **V2),
            "Forecast LP v2, history mean (no forecast)": lambda: ForecastLP(w, h, forecaster="history-mean", **V2),
            "Forecast LP v2, perfect forecast (diagnostic)": lambda: ForecastLP(w, h, forecaster="perfect", future=tr, **V2)}[name]()


def job(args):
    scen, s, only = args
    w0 = E.WSN(seed=s)
    h, tr = TR.generate(scen, w0.pos[:w0.n], s + 1000)
    To = oracle_T(E.WSN(seed=s), tr)
    out = {"oracle": To}
    for m in (only or METHODS):
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
    ap.add_argument("--only", default="", help="|-separated methods to (re)run and merge into results_dynamic.json")
    ap.add_argument("--summarize-only", action="store_true", help="rebuild results_dynamic.md from the JSON")
    ap.add_argument("--scen", default="RNB", help="scenarios to run: any of R, N, B and the robustness scenario D")
    a = ap.parse_args()
    if a.summarize_only:
        raw = json.load(open("results_dynamic.json"))
        summarize({sc: {int(s): v for s, v in raw[sc].items()} for sc in raw}, a.seeds)
        return
    only = a.only.split("|") if a.only else None
    jobs = [(sc, s, only) for sc in a.scen for s in range(a.seeds)]
    res = {sc: {} for sc in a.scen}
    if only or set(a.scen) != set(SCEN):                      # merge into the saved results
        raw = json.load(open("results_dynamic.json"))
        res = {sc: {int(k): v for k, v in raw[sc].items()} for sc in raw}
        for sc in a.scen:
            res.setdefault(sc, {})
    with Pool(a.workers) as p:
        for sc, s, out in p.imap_unordered(job, jobs):
            res[sc][s] = {**res[sc].get(s, {}), **out}
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


def _paired(a, b):
    d = a - b
    p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return f"{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}, p={p:.2g}"


ABLATIONS = [("Forecast LP v2, Holt-Winters (ours)", "Forecast LP, Holt-Winters tuned (ours)", "what the planner fixes add"),
             ("Forecast LP v2, Holt-Winters (ours)", "Forecast LP v2, history mean (no forecast)", "what the forecast adds"),
             ("Forecast LP v2, perfect forecast (diagnostic)", "Forecast LP v2, Holt-Winters (ours)", "room left for a better forecast")]


def summarize(res, seeds):
    methods = [m for m in METHODS if all(m in res[sc][0] for sc in SCEN)] + \
              [m for m in res["R"][0] if m not in METHODS and m != "oracle"]
    extra = [sc for sc in EXTRA_SCEN if sc in res and len(res[sc]) >= seeds]
    allsc = list(SCEN) + extra
    has = {m: [sc for sc in allsc if all(m in res[sc].get(s, {}) for s in range(seeds))] for m in methods}
    L = {m: {sc: np.array([res[sc][s][m] for s in range(seeds)]) for sc in has[m]} for m in methods}
    ref = "Battery Dijkstra (reactive)"
    lines = [f"{seeds} held-out deployments per scenario; L = lifetime to first node death, % of the oracle LP."]
    if extra:
        lines.append("S = equal-weight mean over the three scored scenarios; " +
                     ", ".join(EXTRA_SCEN[sc] for sc in extra) + " is reported beside it, not inside it.")
    lines += ["", "| Method | " + " | ".join(SCEN.values()) + " | Mean S (equal weights) | Worst % |" +
              "".join(f" {EXTRA_SCEN[sc]} |" for sc in extra), "|---|" + "---|" * (len(SCEN) + 2 + len(extra))]
    for m in methods:
        v = [L[m][sc].mean() for sc in SCEN]
        ex = " | ".join(f"{L[m][sc].mean():.1f}" if sc in L[m] else "–" for sc in extra)
        lines.append(f"| {m} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {np.mean(v):.1f} | {min(L[m][sc].min() for sc in SCEN):.1f} |" +
                     (f" {ex} |" if extra else ""))
    lines += ["", f"Paired vs {ref} (wins/ties/losses, Wilcoxon p):", ""]
    for m in methods:
        if m == ref:
            continue
        lines.append(f"- {m}: " + "; ".join(f"{sc}: {_paired(L[m][sc], L[ref][sc])}" for sc in has[m] if sc in L[ref]))
    pairs = [(a, b, why) for a, b, why in ABLATIONS if a in L and b in L]
    if pairs:
        lines += ["", "Ablations (first vs second, wins/ties/losses, Wilcoxon p):", ""]
        for a_, b_, why in pairs:
            cells = [f"{sc}: {L[a_][sc].mean() - L[b_][sc].mean():+.1f} pts, {_paired(L[a_][sc], L[b_][sc])}"
                     for sc in allsc if sc in L[a_] and sc in L[b_]]
            lines.append(f"- {a_} vs {b_} ({why}): " + "; ".join(cells))
    lines += ["", "Break-even forecast accuracy (G = gain when the forecast is right; C = loss when not needed / burst):", ""]
    for m in [m for m in methods if m.startswith("Forecast LP") or m.startswith(("Predictive", "Lifetime", "Price"))]:
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
