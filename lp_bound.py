"""Chang-Tassiulas maximum-lifetime LP: the provable ceiling on first-node-death.

max T subject to, for every node i:
    sum_j f_ij - sum_k f_ki = L * T                (it originates L bits per round)
    sum_j f_ij * etx_bit(i,j) + sum_k f_ki * erx_bit + T * erx(control) <= E0

f_ij are total bits carried on link i -> j over the whole lifetime. Letting flows be
fractional and time-shared is a relaxation of any per-round routing scheme, so T* is an
upper bound no scheme can beat. Solved with HiGHS; ~0.2 s for 100 nodes.
"""
import numpy as np
from scipy.optimize import linprog

from env import WSN, erx


def max_lifetime_T(env, alive=None, energy=None, return_flow=False):
    """T* in rounds. return_flow=True also gives {i: [(next hop, share), ...]} over the
    alive nodes (shares sum to 1), the optimal split used by lp_flow.py."""
    n = env.n
    alive = np.flatnonzero(env.alive if alive is None else alive)
    E = (env.E if energy is None else energy)[alive]
    idx = {node: k for k, node in enumerate(alive)}
    pairs = [(i, j) for i in alive for j in np.append(alive, n) if i != j]
    nv = len(pairs) + 1
    c = np.zeros(nv); c[-1] = -1.0
    A = np.zeros((alive.size, nv)); b = E.astype(float).copy()
    Aeq = np.zeros((alive.size, nv)); beq = np.zeros(alive.size)
    for k, (i, j) in enumerate(pairs):
        A[idx[i], k] += env.etx_bit[i, j]
        if j != n:
            A[idx[j], k] += 50e-9
        Aeq[idx[i], k] += 1.0
        if j != n:
            Aeq[idx[j], k] -= 1.0
    A[:, -1] += float(erx(env.LC))
    Aeq[:, -1] = -float(env.L)
    r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq,
                bounds=[(0, None)] * nv, method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    if not return_flow:
        return float(r.x[-1])
    flow = {}
    for k, (i, j) in enumerate(pairs):
        if r.x[k] > 1e-9:
            flow.setdefault(int(i), []).append((int(j), r.x[k]))
    for i in alive:
        edges = flow.get(int(i), [(n, 1.0)])
        tot = sum(g for _, g in edges)
        flow[int(i)] = [(j, g / tot) for j, g in edges]
    return float(r.x[-1]), flow


if __name__ == "__main__":
    for s in range(3):
        w = WSN(seed=s)
        print(f"seed {s}: ceiling T* = {max_lifetime_T(w):.1f} rounds")
