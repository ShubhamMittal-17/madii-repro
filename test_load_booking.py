"""Regression checks for source-load accounting in coordinated routing."""
import numpy as np

import env as E
import traffic as TR
from predictive import LESTDijkstra, LoadAwareDijkstra, subtree_sum


def check_tree_and_load(w, pol, base, rate):
    p = pol._tree(w, base, rate)
    assert np.array_equal(p, w._sanitize(p)), 'Construction produced an invalid tree'
    expected = subtree_sum(w, p, rate)
    np.testing.assert_allclose(pol.booked_load, expected, rtol=1e-12, atol=1e-12)
    assert np.all(pol.booked_load[~w.alive] == 0), 'Dead sources contributed load'
    return p


def test_assigned_relay_still_books_own_traffic():
    w = E.WSN(n=3, seed=0)
    # Distances enforce processing order 0, 1, 2. Synthetic positive link
    # costs make 0 route through the not-yet-processed source 1.
    w.d2s = np.array([1., 2., 3.])
    w.etx_bit[:] = 1.
    w.etx_bit[0, 1] = 1e-6
    w.etx_bit[1, 3] = 1e-6
    pol = LoadAwareDijkstra(w, np.ones((24,3)), kappa=1., order='near')
    rate = np.array([1., 7., 2.])
    p = check_tree_and_load(w, pol, np.ones(3), rate)
    assert p[0] == 1 and p[1] == 3
    assert pol.booked_load[1] >= 8.


def test_accounting_across_orders_deaths_and_zero_rates():
    for seed in range(3):
        for order in ['near', 'far']:
            w = E.WSN(n=12, seed=seed)
            w.alive[[2,5]] = False
            w.E[[2,5]] = 0.
            rate = np.arange(12,dtype=float) % 4
            rate[~w.alive] = 1000.  # Excluded even with nonzero stored rates.
            pol = LoadAwareDijkstra(w, np.ones((24,12)), kappa=1., order=order)
            check_tree_and_load(w, pol, np.ones(12), rate)


def test_original_deployments_and_lest_decoded_rates():
    for seed in range(3):
        w0 = E.WSN(seed=seed)
        h,tr = TR.generate('R',w0.pos[:w0.n],seed+1000)
        w = E.WSN(seed=seed,traffic=tr)
        pol = LESTDijkstra(w,h,levels='lest',rule='original',band=.05,exact_energy=True)
        for _ in range(5):
            p = pol(w)
            decoded = (1.-pol.L_tab.mid[pol.L_tab.tier])*pol.budget
            np.testing.assert_allclose(pol.booked_load,subtree_sum(w,p,decoded),rtol=1e-12,atol=1e-12)
            w.step(p)


if __name__ == '__main__':
    for test in [test_assigned_relay_still_books_own_traffic,
                 test_accounting_across_orders_deaths_and_zero_rates,
                 test_original_deployments_and_lest_decoded_rates]:
        test()
        print('PASS',test.__name__,flush=True)
