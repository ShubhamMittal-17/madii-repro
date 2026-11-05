"""Run revision experiments; one JSON row per (experiment, condition, scenario, seed, method).

  .venv/bin/python -m revision.run --exp main  --workers 4     # n=100, R/N/B/D, seeds 0-29
  .venv/bin/python -m revision.run --exp grid  --workers 4     # robustness, fresh seeds 2000-2029
  .venv/bin/python -m revision.run --exp kappa --workers 4     # kappa sweep, fresh seeds
  .venv/bin/python -m revision.run --exp lossy --workers 4     # lossy links + contention
  .venv/bin/python -m revision.run --exp tune  --workers 4     # validation seeds 500-511

Score = 100 (F - 1) / T*, with F the round of the first node death and T* the oracle ceiling
for that deployment and traffic (completed death-free rounds over the bound). A run that
reaches the end of the traffic horizon without a death is censored and scored on its observed
rounds; censoring is flagged in the row.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import traffic as TR
from revision.model import Deployment
from revision.oracle import oracle_T
from revision import routers as RT

OUT = Path("results/revision")
TEST_SEEDS = list(range(30))
FRESH_SEEDS = list(range(2000, 2030))
VAL_SEEDS = list(range(500, 512))

GRID_METHODS = ["Min-energy Dijkstra", "Battery-weighted Dijkstra", "Load-balanced SPT (Shan et al.)",
                "Flow augmentation (Chang-Tassiulas)", "Booked-load Dijkstra", "LP planner (every round)"]
CONDITIONS = {                      # robustness grid (M6); main setting is n=100, uniform, centre
    "n50": dict(n=50), "n200": dict(n=200), "sink-corner": dict(sink="corner"),
    "clustered": dict(layout="clustered"), "clustered-corner": dict(layout="clustered", sink="corner"),
}


class TraceDeployment(Deployment):
    """The Intel Lab layout (54 motes, scaled to MADII's field) with the revised accounting."""

    def __init__(self, e0=1.0, traffic=None, accounting="revised", **kw):
        import intel as I
        super().__init__(n=I.N, e0=e0, traffic=traffic, accounting=accounting)
        xy, sink = I.positions()
        self.set_positions(np.vstack([xy, sink[None, :]]))


def make_env(cfg, seed, traffic=None):
    if cfg.get("trace"):
        return TraceDeployment(e0=cfg.get("e0", 1.0), traffic=traffic)
    kw = {k: v for k, v in cfg.items() if k in ("n", "side", "e0", "layout", "sink", "accounting")}
    if cfg.get("links") == "lossy":
        from revision.lossy import LossyDeployment
        return LossyDeployment(seed=seed, traffic=traffic, **kw, **cfg.get("link_kw", {}))
    return Deployment(seed=seed, traffic=traffic, **kw)


def cached_oracle(cfg, scen, seed, tr):
    """Oracle T* depends only on the deployment, accounting and traffic: cache it on disk.
    Lossy runs use the loss-free link energies' ETX-scaled version, so the key includes links."""
    key = json.dumps({k: v for k, v in cfg.items() if k != "link_kw" or v}, sort_keys=True) + f"|{scen}|{seed}"
    d = OUT / "oracle"; d.mkdir(parents=True, exist_ok=True)
    import hashlib
    f = d / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text())["T"]
    T = oracle_T(make_env(cfg, seed), tr)
    f.write_text(json.dumps({"key": key, "T": T}))
    return T


def job(args):
    exp, cond, cfg, scen, seed, methods, params = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    w0 = make_env(cfg, seed)
    if cfg.get("trace"):                     # Intel Lab: seed = start hour of the deployment
        allt = np.load("data/intel_lab/traffic_dT0.5_dL100.npy")
        hist, tr = allt[:seed], allt[seed:]
    else:
        hist, tr = TR.generate(scen, w0.pos[:w0.n], seed + 1000, T=cfg.get("T", 400))
    To = cached_oracle(cfg, scen, seed, tr)
    rows = []
    for m in methods:
        w = make_env(cfg, seed, tr)
        pol = RT.build(m, w, hist, params.get(m))
        F, t0, delivered, generated = None, time.time(), 0.0, 0.0
        while w.round < len(tr):
            info = w.step(pol(w))
            delivered += float(info["b_succ"].sum()) * w.payload_bits / w.L
            generated += float(info["b_succ"].sum() + info["b_fail"].sum()) * w.payload_bits / w.L
            if info["n_dead"] >= 1:
                F = info["round"]
                break
        cens = F is None
        obs = w.round if cens else F - 1
        rows.append(dict(exp=exp, cond=cond, scen=scen, seed=seed, method=m, params=params.get(m),
                         oracle=To, F=F, censored=cens, score=100.0 * obs / To,
                         delivered_kbit=delivered, generated_kbit=generated, sec_per_round=(time.time() - t0) / max(w.round, 1)))
    return rows


def jobs_for(exp):
    J = []
    if exp == "main":
        for sc in "RNBD":
            for s in TEST_SEEDS:
                J.append((exp, "main", {}, sc, s, RT.MAIN, {}))
    elif exp == "grid":
        for cond, cfg in CONDITIONS.items():
            meths = [m for m in GRID_METHODS if not (cond == "n200" and m.startswith("LP planner"))]
            for sc in "RN":
                for s in FRESH_SEEDS:
                    J.append((exp, cond, cfg, sc, s, meths, {}))
    elif exp == "kappa":
        for k in (0.25, 0.5, 1.0, 2.0, 4.0):
            for sc in "RN":
                for s in FRESH_SEEDS:
                    J.append((exp, f"kappa={k}", {}, sc, s, ["Booked-load Dijkstra"], {"Booked-load Dijkstra": {"kappa": k}}))
        for sc in "RN":
            for s in FRESH_SEEDS:
                J.append((exp, "kappa=0", {}, sc, s, ["Battery-weighted Dijkstra"], {}))
    elif exp == "lossy":
        cfg = {"links": "lossy"}
        meths = ["Min-energy Dijkstra", "Battery-weighted Dijkstra", "Flow augmentation (Chang-Tassiulas)",
                 "Booked-load Dijkstra", "LP planner (every round)"]
        for cond, lk in {"lossy": {}, "lossy+contention": {"contention": True}}.items():
            for sc in "RN":
                for s in TEST_SEEDS:
                    J.append((exp, cond, {**cfg, "link_kw": lk}, sc, s, meths, {}))
    elif exp == "tune":
        for x in (1.0, 2.0, 5.0, 20.0, 50.0):
            for H in (1, 6, 24, 96):
                for sc in "RNB":
                    for s in VAL_SEEDS:
                        J.append((exp, f"fa x={x} H={H}", {}, sc, s, ["Flow augmentation (Chang-Tassiulas)"],
                                  {"Flow augmentation (Chang-Tassiulas)": {"x": x, "H": H}}))
        for R in (90.0, 110.0, 125.0, 150.0, 200.0):
            for sc in "RNB":
                for s in VAL_SEEDS:
                    J.append((exp, f"shan R={R}", {}, sc, s, ["Load-balanced SPT (Shan et al.)"],
                              {"Load-balanced SPT (Shan et al.)": {"R": R}}))
    elif exp == "trace":                 # Intel Lab trace-driven traffic, the 27 original test starts
        meths = [m for m in RT.MAIN if "load only" not in m]
        for e0 in (1.0, 1.5):
            for start in range(336, 545, 8):
                J.append((exp, f"intel e0={e0}", {"trace": True, "e0": e0}, "I", start, meths, {}))
    elif exp == "tune2":                 # finer flow-augmentation grid + booked-load kappa, validation only
        for x in (0.5, 1.0):
            for H in (1, 2, 3):
                for K in (5, 10, 30):
                    for sc in "RNB":
                        for s in VAL_SEEDS:
                            J.append((exp, f"fa x={x} H={H} K={K}", {}, sc, s, ["Flow augmentation (Chang-Tassiulas)"],
                                      {"Flow augmentation (Chang-Tassiulas)": {"x": x, "H": H, "K": K}}))
        for k in (0.5, 1.0, 2.0):
            for sc in "RNB":
                for s in VAL_SEEDS:
                    J.append((exp, f"bld kappa={k}", {}, sc, s, ["Booked-load Dijkstra"], {"Booked-load Dijkstra": {"kappa": k}}))
        for sc in "RNB":
            for s in VAL_SEEDS:
                J.append((exp, "battery", {}, sc, s, ["Battery-weighted Dijkstra", "LP planner (every round)"], {}))
    else:
        raise KeyError(exp)
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N jobs (smoke test)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.exp}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["cond"], r["scen"], r["seed"], r["method"], json.dumps(r["params"], sort_keys=True)))
    J = [j for j in jobs_for(a.exp)
         if any((j[1], j[3], j[4], m, json.dumps(j[6].get(m), sort_keys=True)) not in done for m in j[5])]
    if a.limit:
        J = J[:a.limit]
    print(f"{a.exp}: {len(J)} jobs", flush=True)
    t0 = time.time()
    with Pool(a.workers) as p, open(path, "a") as f:
        for k, rows in enumerate(p.imap_unordered(job, J)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 10 == 0:
                r = rows[0]
                print(f"[{k + 1}/{len(J)} {time.time() - t0:.0f}s] {r['cond']} {r['scen']} {r['seed']} "
                      + ", ".join(f"{x['method'][:14]}={x['score']:.1f}" for x in rows), flush=True)


if __name__ == "__main__":
    main()
