"""Real-traffic test on the Intel Berkeley Lab data (see intel.py for how traffic is built).

Deployments: a network is switched on at hour s with full batteries (E0, chosen on the
training days so that battery Dijkstra lives about 4 days) and runs on the real traffic
from s on; the forecasters see every hour before s as history (at least 3 days).
  training deployments  s = 72, 84, ..., 180   (inside the first 14 days)
  test deployments      s = 336, 344, ..., 544 (later days only; run once)
Score: lifetime to first node death as % of the oracle LP for that deployment's traffic.

  .venv/bin/python intel_bench.py --phase tune    # Holt-Winters grid, training deployments
  .venv/bin/python intel_bench.py --phase test    # every method, test deployments
"""
import argparse, json, os
from multiprocessing import Pool

import numpy as np
from scipy.stats import wilcoxon

import intel as I
import dijkstra_rl as D
from lp_oracle import oracle_T
from forecast import FORECASTERS
from predictive import ForecastLP, PredictiveDijkstra, ReactiveLP, StaticLP

E0 = 1.5
DT, DL = 0.5, 100.0
TRAIN_STARTS = list(range(72, 181, 12))
TEST_STARTS = list(range(336, 545, 8))
V2 = dict(resolve_every=1, mu=1e-3)
GRID = [dict(alpha=a, beta=0.0, gamma=g) for a in (0.02, 0.05, 0.1, 0.2) for g in (0.05, 0.1, 0.3)]
TRAFFIC = f"{I.DIR}/traffic_dT{DT}_dL{DL:g}.npy"
_net = None


def traffic():
    if not os.path.exists(TRAFFIC):
        np.save(TRAFFIC, I.traffic(DT, DL))
    return np.load(TRAFFIC)


def _madii():
    global _net
    if _net is None:
        from evaluate import load
        _net = load("checkpoints/madii_v3_best.pt")[0]
    from train import greedy_action
    return lambda w: greedy_action(_net, w)


def methods(hw):
    return {
        "Battery Dijkstra": lambda w, h, tr: D.fast_battery_weighted,
        "Predictive Dijkstra, Holt-Winters": lambda w, h, tr: PredictiveDijkstra(w, h, lam=0.5, H=12, **hw),
        "Static LP (solved once)": lambda w, h, tr: StaticLP(w, h),
        "Reactive LP coordinator": lambda w, h, tr: ReactiveLP(w, h),
        "Forecast LP v1, Holt-Winters": lambda w, h, tr: ForecastLP(w, h, **hw),
        "Forecast LP v2, Holt-Winters (ours)": lambda w, h, tr: ForecastLP(w, h, **hw, **V2),
        "Forecast LP v2, history mean (no forecast)": lambda w, h, tr: ForecastLP(w, h, forecaster="history-mean", **V2),
        "Forecast LP v2, seasonal-naive": lambda w, h, tr: ForecastLP(w, h, forecaster="seasonal-naive", **V2),
        "Forecast LP v2, perfect forecast (diagnostic)": lambda w, h, tr: ForecastLP(w, h, forecaster="perfect", future=tr, **V2),
        "MADII (not retrained)": lambda w, h, tr: _madii(),
    }


def run(args):
    name, hw, s, To = args
    tr = traffic()
    h, fut = tr[:s], tr[s:]
    w = I.IntelWSN(e0=E0, traffic=fut)
    pol = methods(hw)[name](w, h, fut)
    while w.round < len(fut) and w.step(pol(w))["n_dead"] < 1:
        pass
    return name, s, (w.round - 1) / To * 100, w.round >= len(fut)


def oracle(s):
    fut = traffic()[s:]
    return s, oracle_T(I.IntelWSN(e0=E0), fut)


def forecast_error(starts, hw):
    """MAE of each forecaster's next-24-hour mean rate per mote, at every start (packets/hour)."""
    tr = traffic(); out = {}
    for name, kw in (("holt-winters", hw), ("history-mean", {}), ("seasonal-naive", {}), ("persistence", {})):
        err = []
        for s in starts:
            f = FORECASTERS[name](I.N, **kw); f.fit_history(tr[:s])
            err.append(np.abs(f.forecast(24)[0].mean(0) - tr[s:s + 24].mean(0)).mean())
        out[name] = float(np.mean(err))
    return out


def paired(a, b):
    d = a - b
    p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return f"{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}, p={p:.2g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["tune", "test"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--e0", type=float, default=None, help="sensitivity check: another battery size (J)")
    a = ap.parse_args()
    global E0
    suffix = ""
    if a.e0 is not None:
        E0, suffix = a.e0, f"_e{a.e0:g}"
    traffic()
    if a.phase == "tune":
        with Pool(a.workers) as p:
            orc = dict(p.map(oracle, TRAIN_STARTS))
            jobs = [("Forecast LP v2, Holt-Winters (ours)", hw, s, orc[s]) for hw in GRID for s in TRAIN_STARTS]
            jobs += [("Forecast LP v2, history mean (no forecast)", {}, s, orc[s]) for s in TRAIN_STARTS]
            res = p.map(run, jobs)
        rows = []
        for k, hw in enumerate(GRID):
            v = [r[2] for r in res[k * len(TRAIN_STARTS):(k + 1) * len(TRAIN_STARTS)]]
            rows.append((np.mean(v), hw))
            print(f"alpha {hw['alpha']:<5} gamma {hw['gamma']:<5} mean {np.mean(v):5.1f}  worst {min(v):5.1f}")
        hm = [r[2] for r in res[len(GRID) * len(TRAIN_STARTS):]]
        print(f"history mean (no forecast)  mean {np.mean(hm):5.1f}  worst {min(hm):5.1f}")
        best = max(rows, key=lambda r: r[0])[1]
        json.dump({"hw": best, "oracle_train": orc}, open("logs/intel_tune.json", "w"), indent=1)
        print("chosen", best)
        return
    hw = json.load(open("logs/intel_tune.json"))["hw"]
    names = list(methods(hw))
    with Pool(a.workers) as p:
        orc = dict(p.map(oracle, TEST_STARTS))
        res = p.map(run, [(n, hw, s, orc[s]) for n in names for s in TEST_STARTS])
    L = {n: np.array([r[2] for r in res if r[0] == n]) for n in names}
    cens = {n: sum(r[3] for r in res if r[0] == n) for n in names}
    fe = forecast_error(TEST_STARTS, hw)
    json.dump({"hw": hw, "E0": E0, "oracle": orc, "L": {n: L[n].tolist() for n in names}, "censored": cens,
               "forecast_mae": fe}, open(f"results_intel{suffix}.json", "w"), indent=1)
    ref = "Battery Dijkstra"
    lines = [f"# Real traffic: Intel Berkeley Lab ({len(TEST_STARTS)} test deployments, {TEST_STARTS[0] // 24 + 1}-{(TEST_STARTS[-1] + 24) // 24 + 1} days after 28 Feb 2004)", "",
             f"54 real motes (layout scaled x{I.SCALE:g} to a 500 m field), send-on-delta traffic (dT {DT} C, dL {DL:g} lux, "
             f"+1 heartbeat/hour), E0 = {E0} J, Holt-Winters alpha {hw['alpha']}, gamma {hw['gamma']} (tuned on training days). "
             f"L = lifetime to first node death, % of the oracle LP.", "",
             "| Method | Mean L | Worst | Best | vs battery Dijkstra (wins/ties/losses, p) |", "|---|---|---|---|---|"]
    for n in names:
        lines.append(f"| {n} | {L[n].mean():.1f} | {L[n].min():.1f} | {L[n].max():.1f} | {'–' if n == ref else paired(L[n], L[ref])} |")
    lines += ["", "Ablations:", "",
              f"- v2 Holt-Winters vs v2 history mean (what the forecast adds): {L['Forecast LP v2, Holt-Winters (ours)'].mean() - L['Forecast LP v2, history mean (no forecast)'].mean():+.1f} pts, "
              f"{paired(L['Forecast LP v2, Holt-Winters (ours)'], L['Forecast LP v2, history mean (no forecast)'])}",
              f"- v2 vs v1, Holt-Winters (what the planner adds): {L['Forecast LP v2, Holt-Winters (ours)'].mean() - L['Forecast LP v1, Holt-Winters'].mean():+.1f} pts, "
              f"{paired(L['Forecast LP v2, Holt-Winters (ours)'], L['Forecast LP v1, Holt-Winters'])}",
              f"- perfect vs Holt-Winters, v2 (room for a better forecast): {L['Forecast LP v2, perfect forecast (diagnostic)'].mean() - L['Forecast LP v2, Holt-Winters (ours)'].mean():+.1f} pts, "
              f"{paired(L['Forecast LP v2, perfect forecast (diagnostic)'], L['Forecast LP v2, Holt-Winters (ours)'])}",
              "", "Forecast error, next-24-hour mean rate per mote (packets/hour, mean absolute error over test starts): " +
              ", ".join(f"{k} {v:.2f}" for k, v in fe.items()),
              "", "Deployments still alive at the end of the data (censored, scored at the data end): " +
              (", ".join(f"{n} {c}" for n, c in cens.items() if c) or "none")]
    md = "\n".join(lines)
    open(f"results_intel{suffix}.md", "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
