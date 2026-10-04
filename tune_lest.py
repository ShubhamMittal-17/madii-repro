"""LEST-driven Dijkstra on VALIDATION deployments 500-511 (R, N, B): how coarse can the shared
energy/load table be before the gain of load-aware Dijkstra is lost? The snapshot is charged
as control traffic (see predictive.LESTDijkstra).   .venv/bin/python tune_lest.py
"""
import json
from multiprocessing import Pool
import numpy as np
import env as E, traffic as TR, dijkstra_rl as D
from predictive import LESTDijkstra, LoadAwareDijkstra

VAL = range(500, 512)
CFGS = {"battery Dijkstra (exact energy, no load)": None,
        "load-aware, exact values": "exact",
        "LEST load table, 4 tiers, original trigger": dict(levels="lest", rule="original", band=0.05, exact_energy=True),
        "LEST load table, 4 tiers, Schmitt trigger": dict(levels="lest", rule="schmitt", band=0.05, exact_energy=True),
        "LEST load table, 8 levels (band 0.02)": dict(levels=8, band=0.02, exact_energy=True),
        "LEST load table, 16 levels (band 0.01)": dict(levels=16, band=0.01, exact_energy=True)}


def job(a):
    name, sc, s, To = a
    w0 = E.WSN(seed=s); h, tr = TR.generate(sc, w0.pos[:w0.n], s + 1000)
    w = E.WSN(seed=s, traffic=tr); c = CFGS[name]
    pol = D.fast_battery_weighted if c is None else LoadAwareDijkstra(w, h, kappa=1.0, order="near") if c == "exact" else LESTDijkstra(w, h, **c)
    while w.round < len(tr) and w.step(pol(w))["n_dead"] < 1:
        pass
    tab = getattr(pol, "L_tab", None)
    return name, sc, (w.round - 1) / To * 100, w.LC, (tab.changes / max(w.round * w.n, 1)) if tab else float("nan")


if __name__ == "__main__":
    orc = json.load(open("logs/oracle_validation.json"))
    res = {}
    with Pool(4) as p:
        for name, sc, L, lc, rate in p.imap_unordered(job, [(n, sc, s, orc[f"{sc}{s}"]) for n in CFGS for sc in "RNB" for s in VAL]):
            res.setdefault(name, []).append((sc, L, lc, rate))
    print(f"{'config':42s} {'R':>6s} {'N':>6s} {'B':>6s} {'mean':>6s} {'worst':>6s} {'ctrl bits/rd':>12s} {'tier change %':>13s}")
    for n in CFGS:
        r = res[n]; v = {sc: np.array([x[1] for x in r if x[0] == sc]) for sc in "RNB"}
        print(f"{n:42s} {v['R'].mean():6.1f} {v['N'].mean():6.1f} {v['B'].mean():6.1f} {np.mean([x.mean() for x in v.values()]):6.1f} "
              f"{min(x.min() for x in v.values()):6.1f} {r[0][2]:12d} {100*np.nanmean([x[3] for x in r]):13.3f}")
    json.dump(res, open("logs/tune_lest_load.json", "w"), indent=1)
