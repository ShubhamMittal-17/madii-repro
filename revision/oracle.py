"""Lifetime ceilings with sparse constraint matrices (same LPs as lp_bound.py / lp_oracle.py,
which build dense matrices; the sparse form lets the bench reach n = 200 and beyond).

static_T(env, gen):        max T s.t. a flow carries gen_i bits per round from every node
                           within its battery (Chang-Tassiulas; fractional-flow relaxation)
oracle_T(env, traffic):    largest integer T for which a flow carries every node's first-T-
                           rounds traffic within its battery (bisection on T)

Receive energy is charged at E_elec per bit and the advertisement at E_RX(LC) per round,
exactly as env.WSN charges them, so both are upper bounds on any sequence of routing trees.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

E_RX_BIT = 50e-9


def _arcs(env):
    n = env.n
    i, j = np.meshgrid(np.arange(n), np.arange(n + 1), indexing="ij")
    keep = i != j
    return i[keep], j[keep]


def _matrices(env):
    """Energy matrix A (n x m) and flow-conservation matrix Aeq (n x m) over all arcs."""
    n = env.n
    i, j = _arcs(env)
    m = len(i)
    k = np.arange(m)
    relay = j < n
    rows = np.concatenate([i, j[relay]])
    cols = np.concatenate([k, k[relay]])
    A = coo_matrix((np.concatenate([env.etx_bit[i, j], np.full(relay.sum(), E_RX_BIT)]), (rows, cols)), shape=(n, m)).tocsr()
    Aeq = coo_matrix((np.concatenate([np.ones(m), -np.ones(relay.sum())]), (rows, cols)), shape=(n, m)).tocsr()
    return A, Aeq, (i, j)


def _feasible(env, A, Aeq, gen_total, T):
    b = np.full(env.n, env.e0) - T * E_RX_BIT * env.LC
    if (b < 0).any():
        return False
    r = linprog(np.zeros(A.shape[1]), A_ub=A, b_ub=b, A_eq=Aeq, b_eq=gen_total, bounds=(0, None), method="highs")
    return r.status == 0


def oracle_T(env, traffic, hi=None):
    A, Aeq, _ = _matrices(env)
    cum = np.cumsum(np.asarray(traffic, float), axis=0) * env.L
    lo, hi = 0, hi or len(traffic)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _feasible(env, A, Aeq, cum[mid - 1], mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def static_T(env, gen=None):
    """Fractional max lifetime (rounds) with each node generating gen_i bits per round."""
    from scipy.sparse import hstack, csr_matrix
    n = env.n
    A, Aeq, _ = _matrices(env)
    gen = np.full(n, float(env.L)) if gen is None else np.asarray(gen, float)
    A = hstack([A, csr_matrix(np.full((n, 1), E_RX_BIT * env.LC))]).tocsr()
    Aeq = hstack([Aeq, csr_matrix(-gen[:, None])]).tocsr()
    c = np.zeros(A.shape[1]); c[-1] = -1.0
    r = linprog(c, A_ub=A, b_ub=np.full(n, env.e0), A_eq=Aeq, b_eq=np.zeros(n), bounds=(0, None), method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    return float(r.x[-1])
