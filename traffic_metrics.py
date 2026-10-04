"""Beyond lifetime: MADII's own traffic metrics for every router, in MADII's own setting
(100 nodes, 500 m field, one packet per node per round, 30 held-out deployments 0-29).

Per deployment, run to half the nodes dead (HND) or 400 rounds, and record at first node death
(FND, 1%), 10% dead and HND (50%):
  rounds    the round it happened (MADII's SR)
  DDV       data delivered to the sink so far, kbit (MADII's DDV)
  EE        kbit delivered per joule spent (MADII's EE)
plus over the whole run: delivery ratio (delivered / generated), mean hops per packet
(a delay proxy) and the share of packets that had to be resent directly after a relay died.

  .venv/bin/python traffic_metrics.py --workers 4      # writes results_traffic.{json,md}
"""
import argparse, json
from multiprocessing import Pool

import numpy as np

import env as E
import dijkstra_rl as D
import policies as P
from predictive import ForecastLP, LoadAwareDijkstra

BETAS = (0.01, 0.10, 0.50)
_net = None


def _madii():
    global _net
    if _net is None:
        from evaluate import load
        _net = load("checkpoints/madii_v3_best.pt")[0]
    from train import greedy_action
    return lambda w: greedy_action(_net, w)


def routers(w, ones):
    hist = ones[:72]
    return {"Battery Dijkstra": D.fast_battery_weighted,
            "Min-energy Dijkstra": lambda e: P.min_energy(e) if hasattr(P, "min_energy") else D.dijkstra_tree(e, np.ones(e.n)),
            "Load-aware Dijkstra (ours, no LP)": LoadAwareDijkstra(w, hist, kappa=1.0, order="near"),
            "Planner v1 (LP every 4 rounds)": ForecastLP(w, hist, forecaster="history-mean"),
            "Planner v2 (LP every round, ours)": ForecastLP(w, hist, forecaster="history-mean", resolve_every=1, mu=1e-3),
            "MADII (v3 rebuild)": _madii()}


def job(s):
    ones = np.ones((472, 100), int)
    out = {}
    for name in ("Battery Dijkstra", "Min-energy Dijkstra", "Load-aware Dijkstra (ours, no LP)", "Planner v1 (LP every 4 rounds)",
                 "Planner v2 (LP every round, ours)", "MADII (v3 rebuild)"):
        w = E.WSN(seed=s, traffic=ones[72:])
        pol = routers(w, ones)[name]
        n0 = w.n * w.e0
        ddv = gen = retry = 0.0; hops = []; rec = {}
        while w.round < 400:
            p = w._sanitize(pol(w))
            hops.append(float(np.mean(w._depth(p)[w.alive] + 1)))
            info = w.step(p)
            ddv += info["b_succ"].sum(); retry += info["b_retry"].sum()
            gen += info["b_succ"].sum() + info["b_fail"].sum()
            for b in BETAS:
                if b not in rec and info["n_dead"] >= np.ceil(b * w.n):
                    rec[b] = {"rounds": info["round"], "DDV": float(ddv), "EE": float(ddv / max(n0 - w.E.sum(), 1e-12))}
            if info["dead_frac"] >= max(BETAS) or info["n_alive_end"] == 0:
                break
        out[name] = {"at": {str(b): rec.get(b, {"rounds": w.round, "DDV": float(ddv), "EE": float(ddv / max(n0 - w.E.sum(), 1e-12))})
                            for b in BETAS},
                     "delivery": float(ddv / max(gen, 1e-12)), "hops": float(np.mean(hops)), "retry": float(retry / max(ddv, 1e-12))}
    return s, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with Pool(a.workers) as p:
        res = dict(p.map(job, range(a.seeds)))
    json.dump(res, open("results_traffic.json", "w"), indent=1)
    names = list(res[0])
    m = lambda n, f: np.mean([f(res[s][n]) for s in range(a.seeds)])
    lines = [f"# Traffic metrics in MADII's setting ({a.seeds} held-out deployments, constant traffic)", "",
             "| Router | FND round | HND round | Data by FND (kbit) | Data by HND (kbit) | kbit per J (to HND) | Delivery ratio | Hops per packet |",
             "|---|---|---|---|---|---|---|---|"]
    for n in names:
        lines.append(f"| {n} | {m(n, lambda r: r['at']['0.01']['rounds']):.1f} | {m(n, lambda r: r['at']['0.5']['rounds']):.1f} | "
                     f"{m(n, lambda r: r['at']['0.01']['DDV']):.0f} | {m(n, lambda r: r['at']['0.5']['DDV']):.0f} | "
                     f"{m(n, lambda r: r['at']['0.5']['EE']):.0f} | {100 * m(n, lambda r: r['delivery']):.2f}% | {m(n, lambda r: r['hops']):.2f} |")
    lines += ["", "MADII as printed in its paper (same setting): FND 19, HND 89, EE@FND 817, DDV@HND 25708."]
    md = "\n".join(lines)
    open("results_traffic.md", "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
