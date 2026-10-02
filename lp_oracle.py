"""Oracle lifetime ceiling for time-varying traffic (Part 3).

Given the true future traffic k[t, i], the largest T such that a flow exists in which every
node's first-T-rounds traffic reaches the sink and no node exceeds its battery:

    sum_j f_ij - sum_k f_ki = L * sum_{t<T} k[t, i]                 for every node i
    sum_j f_ij e_tx(i,j) + sum_k f_ki e_rx + T * E_RX(L_c) <= E0      f >= 0

Any per-round routing that keeps all nodes alive for T rounds yields link totals that
satisfy these constraints, so T_oracle is an upper bound for every policy, predictive or
not. Feasibility is monotone in T, so we bisect on integer T. With constant traffic this
equals max_lifetime_T (tested).
"""
import numpy as np
from scipy.optimize import linprog

from env import erx


def _feasible(env, gen_total, T):
    n = env.n
    nodes = np.arange(n)
    pairs = [(i, j) for i in nodes for j in np.append(nodes, n) if i != j]
    m = len(pairs)
    A = np.zeros((n, m)); Aeq = np.zeros((n, m))
    for k, (i, j) in enumerate(pairs):
        A[i, k] += env.etx_bit[i, j]; Aeq[i, k] += 1.0
        if j != n:
            A[j, k] += 50e-9; Aeq[j, k] -= 1.0
    b = np.full(n, env.e0) - T * float(erx(env.LC))
    if (b < 0).any():
        return False
    r = linprog(np.zeros(m), A_ub=A, b_ub=b, A_eq=Aeq, b_eq=gen_total, bounds=[(0, None)] * m, method="highs")
    return r.status == 0


def oracle_T(env, traffic, hi=None):
    """Largest integer T with a feasible flow for the first T rounds of `traffic`."""
    cum = np.cumsum(np.asarray(traffic, float), axis=0) * env.L          # cum[t] = bits in rounds 0..t
    lo, hi = 0, hi or len(traffic)
    while lo < hi:                                                       # largest feasible T in [0, hi]
        mid = (lo + hi + 1) // 2
        if _feasible(env, cum[mid - 1], mid):
            lo = mid
        else:
            hi = mid - 1
    return lo
