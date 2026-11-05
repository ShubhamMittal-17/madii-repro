"""Checks for the revision bench.  .venv/bin/python test_revision.py"""
import numpy as np

import env as E
import lp_bound
import lp_oracle
import traffic as TR
from predictive import LoadAwareDijkstra
from revision import routers as RT
from revision.lossy import LossyDeployment
from revision.model import Deployment, advert_bits
from revision.oracle import oracle_T, static_T


def _is_tree(w, p):
    for i in np.flatnonzero(w.alive):
        u, k = i, 0
        while u != w.n:
            u = p[u]; k += 1
            assert k <= w.n, "cycle"


def test_legacy_matches_env():
    a, b = E.WSN(seed=3), Deployment(seed=3, accounting="legacy")
    assert np.allclose(a.pos, b.pos) and a.L == b.L and a.LC == b.LC


def test_revised_accounting():
    w = Deployment(n=100)
    assert w.L == 4016 and w.LC == advert_bits(100) == 800


def test_sparse_oracle_matches_dense():
    w = E.WSN(seed=1)
    _, tr = TR.generate("R", w.pos[:100], 1001)
    assert oracle_T(Deployment(seed=1, accounting="legacy"), tr) == lp_oracle.oracle_T(E.WSN(seed=1), tr)
    assert abs(static_T(Deployment(seed=1, accounting="legacy")) - lp_bound.max_lifetime_T(E.WSN(seed=1))) < 1e-6


def test_routers_return_trees():
    for layout, sink in (("uniform", "center"), ("clustered", "corner")):
        w = Deployment(seed=2, layout=layout, sink=sink, traffic=np.ones((5, 100), int))
        h = np.ones((24, 100), int)
        for m in RT.MAIN:
            if m.startswith("LP"):
                continue
            _is_tree(w, w._sanitize(RT.build(m, w, h)(w)))


def test_booking_counts_every_source():
    w = Deployment(seed=0, traffic=np.ones((5, 100), int))
    r = LoadAwareDijkstra(w, np.ones((24, 100), int), kappa=1.0, order="near")
    p = r(w)
    # booked load of a node = rates of all sources whose path crosses it (itself included)
    expect = np.zeros(100)
    for i in range(100):
        u = i
        while u != 100:
            expect[u] += 1.0; u = p[u]
    assert np.allclose(r.booked_load, expect)


def test_lossy_reduces_to_ideal_when_links_perfect():
    w = LossyDeployment(seed=0, traffic=np.ones((5, 100), int))
    w.prr[:] = 1.0
    w.etx_bit = w.radio_bit.copy()
    v = Deployment(seed=0, traffic=np.ones((5, 100), int))
    p = RT.build("Battery-weighted Dijkstra", v, None)(v)
    a, b = w.step(p), v.step(p)
    assert np.allclose(w.E, v.E) and np.allclose(a["b_succ"], b["b_succ"])


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
