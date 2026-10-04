"""Tune the EBR-DA-style baseline on VALIDATION deployments 500-511 (R, N, B), same budget and
protocol as load-aware Dijkstra.   .venv/bin/python tune_ebrda.py
"""
import json, itertools
from multiprocessing import Pool
import numpy as np
import env as E, traffic as TR, dijkstra_rl as D
from predictive import EBRDA

VAL = range(500, 512)
GRID = [None] + [dict(a=a, c=c) for a, c in itertools.product((0.5, 0.75, 1.0), (30.0, 100.0, 300.0, 1000.0))]


def job(x):
    k, sc, s, To = x
    cfg = GRID[k]
    w0 = E.WSN(seed=s); h, tr = TR.generate(sc, w0.pos[:w0.n], s + 1000)
    w = E.WSN(seed=s, traffic=tr)
    pol = D.fast_battery_weighted if cfg is None else EBRDA(w, h, **cfg)
    while w.round < len(tr) and w.step(pol(w))["n_dead"] < 1:
        pass
    return k, sc, (w.round - 1) / To * 100


if __name__ == "__main__":
    orc = json.load(open("logs/oracle_validation.json"))
    res = {}
    with Pool(4) as p:
        for k, sc, L in p.imap_unordered(job, [(k, sc, s, orc[f"{sc}{s}"]) for k in range(len(GRID)) for sc in "RNB" for s in VAL]):
            res.setdefault(k, {}).setdefault(sc, []).append(L)
    rows = []
    for k, cfg in enumerate(GRID):
        v = {sc: np.mean(res[k][sc]) for sc in "RNB"}; m = np.mean(list(v.values()))
        worst = min(min(res[k][sc]) for sc in "RNB")
        rows.append((m, k))
        name = "battery Dijkstra" if cfg is None else f"a={cfg['a']:<4} c={cfg['c']:<4}"
        print(f"{name:20s} R {v['R']:5.1f} N {v['N']:5.1f} B {v['B']:5.1f} mean {m:5.1f} worst {worst:5.1f}")
    best = max(r for r in rows if GRID[r[1]] is not None)
    print("chosen", GRID[best[1]], f"mean {best[0]:.1f}")
    json.dump({"chosen": GRID[best[1]], "res": {str(k): v for k, v in res.items()}}, open("logs/tune_ebrda.json", "w"), indent=1)
