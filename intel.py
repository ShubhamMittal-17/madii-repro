"""Intel Berkeley Research Lab data (54 motes, 28 Feb - 5 Apr 2004) as a real-traffic bench.

Readings: data/intel_lab/data.txt.gz (from the repo's `intel_data` release), parsed once into
data/intel_lab/readings.npz by `.venv/bin/python intel.py --parse`. Positions: mote_locs.txt.

How real readings become traffic (one round = one hour, as in traffic.py):
  * cleaning: drop readings from motes at < 2.4 V or with temperature outside 0-50 C (a dying
    mote reports 122 C; these coincide with low voltage in 99.999% of cases);
  * event-driven reporting (send-on-delta): a mote sends a packet whenever its temperature
    or light has moved more than dT / dL since the last value it reported, plus one
    heartbeat packet per hour. Lights, sun and occupancy then give real daily and weekly
    patterns; hours with no clean readings get the heartbeat only.
  * the 31 days 28 Feb - 29 Mar are used (after that most motes have died). Thresholds and
    every tuned setting come from the first 14 days (TRAIN_END); later days are the test.

Topology: the real mote layout, scaled from the 40 m x 31 m lab to MADII's 500 m field
(factor SCALE). At lab scale the first-order radio model makes the direct link to the sink
almost always cheapest, so there would be no routing problem to study. Sink at the centre of
the layout.
"""
import argparse, gzip, datetime as dt
import numpy as np

import env as E

DIR = "data/intel_lab"
DAY0 = dt.datetime(2004, 2, 28)
HOURS = 31 * 24
TRAIN_END = 14 * 24
N = 54
SCALE = 12.0


def parse():
    M, T, TE, LI, V = [], [], [], [], []
    with gzip.open(f"{DIR}/data.txt.gz", "rt") as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            try:
                m = int(p[3]); te, li, v = float(p[4]), float(p[6]), float(p[7])
                stamp = p[0] + " " + (p[1] if "." in p[1] else p[1] + ".0")
                ts = (dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f") - DAY0).total_seconds()
            except ValueError:
                continue
            if 1 <= m <= N:
                M.append(m); T.append(ts); TE.append(te); LI.append(li); V.append(v)
    M, T, TE, LI, V = map(np.array, (M, T, TE, LI, V))
    o = np.lexsort((T, M))
    np.savez_compressed(f"{DIR}/readings.npz", mote=M[o].astype(np.int16), t=T[o], temp=TE[o].astype(np.float32),
                        light=LI[o].astype(np.float32), volt=V[o].astype(np.float32))


def readings():
    r = np.load(f"{DIR}/readings.npz")
    ok = (r["volt"] >= 2.4) & (r["temp"] >= 0) & (r["temp"] <= 50) & (r["t"] < HOURS * 3600)
    return r["mote"][ok] - 1, r["t"][ok], r["temp"][ok].astype(float), r["light"][ok].astype(float)


def traffic(dT=0.5, dL=50.0, heartbeat=1, data=None):
    """(HOURS, N) packets per mote per hour from send-on-delta reporting."""
    mote, t, te, li = data if data is not None else readings()
    out = np.full((HOURS, N), heartbeat, int)
    for m in range(N):
        idx = np.flatnonzero(mote == m)
        if idx.size == 0:
            continue
        lt, ll = te[idx[0]], li[idx[0]]
        for k in idx[1:]:
            if abs(te[k] - lt) > dT or abs(li[k] - ll) > dL:
                out[int(t[k] // 3600), m] += 1
                lt, ll = te[k], li[k]
    return out


def positions():
    xy = np.loadtxt(f"{DIR}/mote_locs.txt")[:, 1:3] * SCALE
    return xy, (xy.min(0) + xy.max(0)) / 2


class IntelWSN(E.WSN):
    """MADII's energy model on the (scaled) real Intel Lab layout."""
    def __init__(self, e0, traffic=None, **kw):
        super().__init__(n=N, e0=e0, traffic=traffic, **kw)
        xy, sink = positions()
        self.pos = np.vstack([xy, sink[None, :]])
        self.d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
        self.d2s = self.d[:N, N]
        self.etx_bit = E.etx(1.0, self.d)
        self.reset()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", action="store_true")
    if ap.parse_args().parse:
        parse()
