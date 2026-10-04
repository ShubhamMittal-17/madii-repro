"""Tune the forecast-LP planner on VALIDATION deployments only (500-511; the held-out test
deployments are 0-29 and are never touched here). Scenarios R, N, B as in compare_dynamic.py.

Each config is a dict of ForecastLP options plus Holt-Winters settings; the score is lifetime
to first node death as % of the oracle LP, and the cost is LP solves per round.

  .venv/bin/python tune_forecast_lp.py --configs base,bits,mu3 --workers 4
Results are merged into logs/tune_forecast_lp.json and printed as a table.
"""
import argparse, json, os, time
from multiprocessing import Pool

import numpy as np

import env as E
import traffic as TR
from lp_oracle import oracle_T
from predictive import ForecastLP

VAL = range(500, 512)
SCEN = ("R", "N", "B")
HW = dict(alpha=0.05, beta=0.0, gamma=0.1)
CONFIGS = {
    "base":        dict(),
    "bits":        dict(split="bits"),
    "mu3":         dict(mu=1e-3),
    "mu2":         dict(mu=1e-2),
    "re2":         dict(resolve_every=2),
    "re1":         dict(resolve_every=1),
    "re8":         dict(resolve_every=8),
    "H12":         dict(H=12),
    "H48":         dict(H=48),
    "Hlife":       dict(H="life"),
    "event3":      dict(event=3.0),
    "event2":      dict(event=2.0),
    "a02g05":      dict(alpha=0.02, gamma=0.05),
    "a02g10":      dict(alpha=0.02, gamma=0.1),
    "a05g05":      dict(alpha=0.05, gamma=0.05),
    "a05g02":      dict(alpha=0.05, gamma=0.02),
    "a10g05":      dict(alpha=0.10, gamma=0.05),
    "perfect":     dict(forecaster="perfect"),
}
OUT = "logs/tune_forecast_lp.json"
ORACLE = "logs/oracle_validation.json"


def oracle(scen, s):
    w0 = E.WSN(seed=s)
    _, tr = TR.generate(scen, w0.pos[:w0.n], s + 1000)
    return oracle_T(E.WSN(seed=s), tr)


def run(args):
    name, cfg, scen, s, To = args
    w0 = E.WSN(seed=s)
    h, tr = TR.generate(scen, w0.pos[:w0.n], s + 1000)
    kw = {**HW, **cfg}
    if kw.get("forecaster") == "perfect":
        kw = {k: v for k, v in kw.items() if k not in HW}
        kw["future"] = tr
    w = E.WSN(seed=s, traffic=tr)
    t0 = time.time()
    pol = ForecastLP(w, h, **kw)
    while w.round < len(tr) and w.step(pol(w))["n_dead"] < 1:
        pass
    return name, scen, s, (w.round - 1) / To * 100, pol.n_solves / max(w.round, 1), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extra", default="", help='JSON {name: config} to try beyond CONFIGS')
    a = ap.parse_args()
    cfgs = {**CONFIGS, **(json.loads(a.extra) if a.extra else {})}
    names = [c for c in a.configs.split(",") if c]
    orc = json.load(open(ORACLE)) if os.path.exists(ORACLE) else {}
    need = [(sc, s) for sc in SCEN for s in VAL if f"{sc}{s}" not in orc]
    if need:
        with Pool(a.workers) as p:
            for (sc, s), T in zip(need, p.starmap(oracle, need)):
                orc[f"{sc}{s}"] = T
        json.dump(orc, open(ORACLE, "w"), indent=1)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    jobs = [(n, cfgs[n], sc, s, orc[f"{sc}{s}"]) for n in names for sc in SCEN for s in VAL]
    with Pool(a.workers) as p:
        for n, sc, s, L, sol, sec in p.imap_unordered(run, jobs):
            res.setdefault(n, {"config": cfgs[n]}).setdefault(sc, {})[str(s)] = [L, sol, sec]
    json.dump(res, open(OUT, "w"), indent=1)
    base = res.get("base")
    print(f"{'config':10s} {'R':>6s} {'N':>6s} {'B':>6s} {'mean':>6s} {'worst':>6s} {'d mean':>7s} {'solves/rd':>9s} {'s/run':>6s}")
    for n in res:
        if not all(sc in res[n] and len(res[n][sc]) == len(VAL) for sc in SCEN):
            continue
        v = {sc: np.array([res[n][sc][str(s)][0] for s in VAL]) for sc in SCEN}
        m = np.mean([v[sc].mean() for sc in SCEN])
        d = m - np.mean([np.mean([base[sc][str(s)][0] for s in VAL]) for sc in SCEN]) if base else float("nan")
        sol = np.mean([res[n][sc][str(s)][1] for sc in SCEN for s in VAL])
        sec = np.mean([res[n][sc][str(s)][2] for sc in SCEN for s in VAL])
        print(f"{n:10s} {v['R'].mean():6.1f} {v['N'].mean():6.1f} {v['B'].mean():6.1f} {m:6.2f} "
              f"{min(x.min() for x in v.values()):6.1f} {d:+7.2f} {sol:9.3f} {sec:6.1f}")


if __name__ == "__main__":
    main()
