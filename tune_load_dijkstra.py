"""Tune load-aware Dijkstra on VALIDATION deployments 500-511 only (test deployments 0-29
untouched), scenarios R, N, B; score = lifetime to first node death, % of the oracle LP.

  .venv/bin/python tune_load_dijkstra.py --workers 4
"""
import argparse, json, itertools
from multiprocessing import Pool

import numpy as np

import env as E
import traffic as TR
import dijkstra_rl as D
from predictive import LoadAwareDijkstra

VAL = range(500, 512)
HW = dict(alpha=0.05, beta=0.0, gamma=0.1)
ORACLE = "logs/oracle_validation.json"


def job(args):
    cfg, sc, s, To = args
    w0 = E.WSN(seed=s)
    h, tr = TR.generate(sc, w0.pos[:w0.n], s + 1000)
    w = E.WSN(seed=s, traffic=tr)
    pol = D.fast_battery_weighted if cfg is None else LoadAwareDijkstra(w, h, **cfg)
    while w.round < len(tr) and w.step(pol(w))["n_dead"] < 1:
        pass
    return json.dumps(cfg), sc, (w.round - 1) / To * 100, getattr(pol, "n_solves", 0) / max(w.round, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kappa", default="0.03,0.1,0.3,1")
    ap.add_argument("--order", default="far,near")
    a = ap.parse_args()
    orc = json.load(open(ORACLE))
    cfgs = [None] + [dict(kappa=float(k), order=o) for k, o in itertools.product(a.kappa.split(","), a.order.split(","))]
    jobs = [(cfg, sc, s, orc[f"{sc}{s}"]) for cfg in cfgs for sc in "RNB" for s in VAL]
    res = {}
    with Pool(a.workers) as p:
        for key, sc, L, _ in p.imap_unordered(job, jobs):
            res.setdefault(key, {}).setdefault(sc, []).append(L)
    print(f"{'config':24s} {'R':>6s} {'N':>6s} {'B':>6s} {'mean':>6s} {'worst':>6s}")
    for cfg in cfgs:
        v = {sc: np.array(res[json.dumps(cfg)][sc]) for sc in "RNB"}
        name = "battery Dijkstra" if cfg is None else f"kappa={cfg['kappa']:g} {cfg['order']}"
        print(f"{name:24s} {v['R'].mean():6.1f} {v['N'].mean():6.1f} {v['B'].mean():6.1f} "
              f"{np.mean([x.mean() for x in v.values()]):6.1f} {min(x.min() for x in v.values()):6.1f}")
    json.dump(res, open("logs/tune_load_dijkstra.json", "w"), indent=1)


if __name__ == "__main__":
    main()
