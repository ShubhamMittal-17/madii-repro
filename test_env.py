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


def test_traffic_none_is_one_packet_per_node():
    """An all-ones traffic array reproduces the paper's setting exactly."""
    import traffic as TR
    w0 = E.WSN(seed=6); _, ones = TR.generate("N", w0.pos[:w0.n], 0)
    a = E.run_episode(E.WSN(seed=6), D.fast_battery_weighted, (0.01, 0.5))
    b = E.run_episode(E.WSN(seed=6, traffic=ones), D.fast_battery_weighted, (0.01, 0.5))
    assert a == b


def test_predictive_lambda0_is_battery_dijkstra():
    import traffic as TR
    from predictive import PredictiveDijkstra
    w0 = E.WSN(seed=7); h, tr = TR.generate("R", w0.pos[:w0.n], 7)
    w1, w2 = E.WSN(seed=7, traffic=tr), E.WSN(seed=7, traffic=tr)
    pol = PredictiveDijkstra(w1, h, lam=0)
    for _ in range(40):
        a, c = pol(w1), D.fast_battery_weighted(w2)
        assert (a == c).all()
        w1.step(a); w2.step(c)


def test_oracle_matches_static_ceiling_and_bounds_policies():
    import traffic as TR
    from lp_oracle import oracle_T
    from predictive import ForecastLP
    w0 = E.WSN(seed=8)
    _, ones = TR.generate("N", w0.pos[:w0.n], 0)
    assert oracle_T(E.WSN(seed=8), ones) == int(max_lifetime_T(E.WSN(seed=8)))
    h, tr = TR.generate("B", w0.pos[:w0.n], 8)
    t = oracle_T(E.WSN(seed=8), tr)
    for mk in (lambda w: D.fast_battery_weighted, lambda w: ForecastLP(w, h)):
        w = E.WSN(seed=8, traffic=tr); pol = mk(w)
        while w.step(pol(w))["n_dead"] < 1:
            pass
        assert w.round - 1 <= t, (w.round, t)


def test_forecasters_see_no_future():
    """A forecaster's output at round t must not change if traffic after t changes."""
    import traffic as TR
    from predictive import _Observer
    w0 = E.WSN(seed=9); h, tr = TR.generate("R", w0.pos[:w0.n], 9)
    tr2 = tr.copy(); tr2[30:] = 0
    for name in ("holt-winters", "seasonal-naive", "persistence"):
        o1, o2 = _Observer(100, h, name), _Observer(100, h, name)
        w1, w2 = E.WSN(seed=9, traffic=tr), E.WSN(seed=9, traffic=tr2)
        w1.round = w2.round = 30
        o1.catch_up(w1); o2.catch_up(w2)
        assert np.allclose(o1.f.forecast(24)[0], o2.f.forecast(24)[0])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}", flush=True); passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}", flush=True)
    print(f"\n{passed}/{len(tests)} passed")
