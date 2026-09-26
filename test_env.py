"""Unit checks for this bench, ported from phase1/test_energy.py plus checks for the new
code (fast Dijkstra, RL-Dijkstra warm start, LP-flow, ceiling). Run: .venv/bin/python test_env.py"""
import numpy as np

import env as E
import policies as P
import dijkstra_rl as D
from lp_bound import max_lifetime_T
from lp_flow import LPFlowAdaptive


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_d0():
    assert approx(E.D0, np.sqrt(10e-12 / 0.0013e-12)) and 87.0 < E.D0 < 88.0


def test_etx_free_space():
    d = 50.0
    assert approx(float(E.etx(4000, d)), 4000 * E.E_ELEC + 4000 * E.EPS_FS * d ** 2)


def test_etx_multipath():
    d = 180.0
    assert approx(float(E.etx(4000, d)), 4000 * E.E_ELEC + 4000 * E.EPS_AMP * d ** 4)


def test_erx():
    assert approx(float(E.erx(4000)), 4000 * E.E_ELEC)


def test_round_conserves_and_delivers():
    """All-direct round: energy drops by direct sends + RouteADV receive (R4); all delivered."""
    w = E.WSN(n=20, seed=1)
    before = w.E.sum()
    expect = float(np.sum(E.etx(w.L, w.d2s))) + w.n * float(E.erx(w.LC))
    info = w.step(P.direct(w))
    assert approx(before - w.E.sum(), expect, tol=1e-6)
    assert info["b_succ"].sum() == w.n * w.L / 1000.0
    assert info["b_retry"].sum() == 0 and info["b_fail"].sum() == 0


def test_relay_costs_more_for_relay_node():
    w = E.WSN(n=3, seed=2)
    before = w.E.copy()
    w.step(np.array([1, 3, 3]))
    assert (before - w.E)[1] > float(E.etx(w.L, w.d2s[1])) + float(E.erx(w.LC))


def test_fast_dijkstra_matches_reference():
    w = E.WSN(seed=4)
    for _ in range(60):
        a = P.battery_weighted(w)
        assert (a == D.fast_battery_weighted(w)).all()
        w.step(a)


def test_rl_dijkstra_warm_start_is_battery_dijkstra():
    w, pol = E.WSN(seed=5), D.Policy(seed=3)
    for _ in range(60):
        a = P.battery_weighted(w)
        assert (a == pol(w)).all()
        w.step(a)


def test_no_policy_beats_ceiling():
    """FND - 1 full rounds survived can never exceed the LP ceiling T*."""
    for s in (0, 1):
        t = max_lifetime_T(E.WSN(seed=s))
        for pol in (D.fast_battery_weighted, D.load("checkpoints/dijkstra_rl.npz"), None):
            w = E.WSN(seed=s)
            fnd = E.run_episode(w, pol or LPFlowAdaptive(w), (0.01,))[0.01]["SR"]
            assert fnd - 1 <= t + 1e-6, (s, fnd, t)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}", flush=True); passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}", flush=True)
    print(f"\n{passed}/{len(tests)} passed")
