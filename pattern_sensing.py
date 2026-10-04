"""Does prediction sense the traffic pattern, and does the router act on it?

Sensing (forecaster only): one round before each evening surge (hour 16), rank nodes by the
Holt-Winters forecast for the next 6 rounds; precision of the top 20% against the true
hotspot set, and the hour at which the forecast puts the hotspot peak.

Acting (router): relay packets and energy spent by hotspot nodes, split into the 5 rounds
before the surge (hours 12-16), the surge itself (17-22) and the rest of the day; how often
the first node to die is a hotspot node; and route churn (share of nodes changing next hop).

30 held-out deployments, scenarios R (daily surge) and B (surge + bursts), same seeds as
compare_dynamic.py.   .venv/bin/python pattern_sensing.py --workers 4
"""
import argparse, json
from multiprocessing import Pool

import numpy as np

import env as E, traffic as TR, dijkstra_rl as D
from forecast import HoltWinters
from predictive import PredictiveDijkstra, ForecastLP, subtree_sum
from compare_dynamic import HW_TUNED

P, PEAK0, PEAKN = 24, 17, 6
PRE = range(12, 17)
SURGE = range(PEAK0, PEAK0 + PEAKN)
ROUTERS = {"Battery Dijkstra": lambda w, h: D.fast_battery_weighted,
           "Predictive Dijkstra (HW)": lambda w, h: PredictiveDijkstra(w, h, lam=0.5, H=12),
           "Forecast LP (HW, ours)": lambda w, h: ForecastLP(w, h, **HW_TUNED)}


def sensing(h, tr, hot):
    hw = HoltWinters(tr.shape[1], **HW_TUNED); hw.fit_history(h)
    prec, peak_err = [], []
    for t in range(len(tr)):
        if t % P == PEAK0 - 1:
            m = hw.forecast(P)[0]
            top = np.argsort(-m[1:1 + PEAKN].mean(0))[:len(hot)]
            prec.append(len(set(top) & set(hot)) / len(hot))
            peak_hour = (t + 1 + int(np.argmax(m[:, hot].mean(1)))) % P
            peak_err.append(min(abs(peak_hour - h_) for h_ in SURGE))
        hw.update(tr[t])
    return float(np.mean(prec)), float(np.mean(peak_err))


def acting(name, s, h, tr, hot):
    w = E.WSN(seed=s, traffic=tr); pol = ROUTERS[name](w, h)
    ishot = np.zeros(w.n, bool); ishot[hot] = True
    acc = {k: [] for k in ("relay_pre", "relay_surge", "relay_other", "energy_pre", "energy_surge", "energy_other")}
    churn, prev = [], None
    while w.round < len(tr):
        a = pol(w); t = w.round; hour = t % P
        p = w._sanitize(a)
        own = np.where(w.alive, tr[t], 0)
        relay = subtree_sum(w, p, own) - own
        if prev is not None:
            churn.append(float(np.mean(p[w.alive] != prev[w.alive])))
        prev = p.copy()
        e0 = w.E.copy()
        info = w.step(a)
        phase = "pre" if hour in PRE else "surge" if hour in SURGE else "other"
        acc["relay_" + phase].append(relay[ishot].sum() / max(relay.sum(), 1e-9))
        spent = e0 - w.E
        acc["energy_" + phase].append(spent[ishot].sum() / max(spent.sum(), 1e-12))
        if info["n_dead"] >= 1:
            break
    first_dead = int(np.flatnonzero(~w.alive)[0]) if (~w.alive).any() else -1
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
    out.update(churn=float(np.mean(churn)), first_dead_hot=float(first_dead in set(hot)), fnd=w.round)
    return out


def job(args):
    scen, s = args
    w0 = E.WSN(seed=s)
    h, tr = TR.generate(scen, w0.pos[:w0.n], s + 1000)
    hot = TR._hotspot(w0.pos[:w0.n], np.random.default_rng(s + 1000))
    out = {"sense": sensing(h, tr, hot)}
    for name in ROUTERS:
        out[name] = acting(name, s, h, tr, hot)
    return scen, s, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    res = {"R": {}, "B": {}}
    with Pool(a.workers) as p:
        for scen, s, out in p.imap_unordered(job, [(sc, s) for sc in res for s in range(a.seeds)]):
            res[scen][s] = out
    json.dump(res, open("results_pattern.json", "w"), indent=1)
    lines = ["# Does prediction sense the pattern and act on it?", "",
             f"{a.seeds} held-out deployments; hotspot = 20% of nodes. Shares are the hotspot nodes' share of all relayed packets / energy spent.", ""]
    for scen, title in (("R", "Daily surge"), ("B", "Surge + bursts")):
        r = [res[scen][s] for s in range(a.seeds)]
        lines += [f"## {title}", "",
                  f"Sensing: top-20% forecast nodes before the surge are true hotspot nodes {100 * np.mean([x['sense'][0] for x in r]):.0f}% of the time; "
                  f"forecast peak is off by {np.mean([x['sense'][1] for x in r]):.2f} hours on average.", "",
                  "| Router | Rounds to FND | Hotspot relay share: pre-surge | surge | rest | Hotspot energy share: pre | surge | rest | First death is a hotspot node | Route churn / round |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name in ROUTERS:
            m = lambda k: np.nanmean([x[name][k] for x in r])
            lines.append(f"| {name} | {m('fnd'):.1f} | {100 * m('relay_pre'):.1f}% | {100 * m('relay_surge'):.1f}% | {100 * m('relay_other'):.1f}% | "
                         f"{100 * m('energy_pre'):.1f}% | {100 * m('energy_surge'):.1f}% | {100 * m('energy_other'):.1f}% | "
                         f"{100 * m('first_dead_hot'):.0f}% | {100 * m('churn'):.1f}% |")
        lines.append("")
    open("results_pattern.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
