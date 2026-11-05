"""Wall-clock cost of one routing decision at the sink as the network grows (review M1, M6).

For n in {50, 100, 200, 400, 800} (uniform field scaled to keep the density of MADII's
setting), times one round of each router on a fresh deployment and one sparse LP solve of the
max-lifetime problem (the work the LP planner repeats every round). Single thread, median of
`reps` runs. Writes results/revision/scaling.json.

  OMP_NUM_THREADS=1 .venv/bin/python -m revision.scaling
"""
import json
import sys
import time

import numpy as np

from revision import routers as RT
from revision.model import Deployment
from revision.oracle import static_T

SIZES = (50, 100, 200, 400, 800)
ROUTERS = ["Battery-weighted Dijkstra", "Booked-load Dijkstra", "Flow augmentation (Chang-Tassiulas)"]


def timed(f, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return float(np.median(t))


def main():
    out = []
    max_lp = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    for n in SIZES:
        side = 500.0 * np.sqrt(n / 100)                     # constant density
        hist = np.ones((24, n), int)
        row = {"n": n, "side": side}
        for m in ROUTERS:
            w = Deployment(n=n, side=side, seed=0, traffic=np.ones((10, n), int))
            pol = RT.build(m, w, hist)
            row[m] = timed(lambda: pol(w), 3 if n <= 200 else 1)
        if n <= max_lp:
            w = Deployment(n=n, side=side, seed=0)
            row["LP solve (sparse HiGHS)"] = timed(lambda: static_T(w), 1)
        out.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        json.dump(out, open("results/revision/scaling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
