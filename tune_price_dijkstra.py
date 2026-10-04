"""Tune price-guided Dijkstra on VALIDATION deployments 500-511 only (test deployments 0-29
untouched), scenarios R, N, B; score = lifetime to first node death, % of the oracle LP.

  .venv/bin/python tune_price_dijkstra.py --workers 4
"""
import argparse, json, itertools
from multiprocessing import Pool

import numpy as np

import env as E
import traffic as TR
import dijkstra_rl as D
from predictive import PriceDijkstra

VAL = range(500, 512)
HW = dict(alpha=0.05, beta=0.0, gamma=0.1)
ORACLE = "logs/oracle_validation.json"


def job(args):
    cfg, sc, s, To = args
    w0 = E.WSN(seed=s)
    h, tr = TR.generate(sc, w0.pos[:w0.n], s + 1000)
    w = E.WSN(seed=s, traffic=tr)
    pol = D.fast_battery_weighted if cfg is None else PriceDijkstra(w, h, **cfg, **HW)
    while w.round < len(tr) and w.step(pol(w))["n_dead"] < 1:
        pass
    return json.dumps(cfg), sc, (w.round - 1) / To * 100, getattr(pol, "n_solves", 0) / max(w.round, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--c", default="0.1,0.3,1,3,10")
    ap.add_argument("--every", default="24,6")
    ap.add_argument("--rx", default="sender")
    ap.add_argument("--eta", default="0")
    a = ap.parse_args()
    orc = json.load(open(ORACLE))
    cfgs = [None] + [dict(c=float(c), every=int(e), rx=r, eta=float(h)) for c, e, r, h in
                     itertools.product(a.c.split(","), a.every.split(","), a.rx.split(","), a.eta.split(","))]
    jobs = [(cfg, sc, s, orc[f"{sc}{s}"]) for cfg in cfgs for sc in "RNB" for s in VAL]
    res = {}
    with Pool(a.workers) as p:
        for key, sc, L, sol in p.imap_unordered(job, jobs):
            res.setdefault(key, {}).setdefault(sc, []).append((L, sol))
    print(f"{'config':28s} {'R':>6s} {'N':>6s} {'B':>6s} {'mean':>6s} {'worst':>6s} {'LP/round':>8s}")
    for cfg in cfgs:
        r = res[json.dumps(cfg)]
        v = {sc: np.array([x[0] for x in r[sc]]) for sc in "RNB"}
        sol = np.mean([x[1] for sc in "RNB" for x in r[sc]])
        name = "battery Dijkstra" if cfg is None else f"c={cfg['c']:g} ev={cfg['every']} {cfg['rx'][:3]} eta={cfg['eta']:g}"
        print(f"{name:28s} {v['R'].mean():6.1f} {v['N'].mean():6.1f} {v['B'].mean():6.1f} "
              f"{np.mean([x.mean() for x in v.values()]):6.1f} {min(x.min() for x in v.values()):6.1f} {sol:8.3f}")
    json.dump(res, open(f"logs/tune_price_dijkstra_{a.rx.replace(',', '_')}_eta{a.eta.replace(',', '_')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
