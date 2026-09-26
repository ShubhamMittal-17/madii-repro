"""The clustering experts MADII imitates: FCM, HBA and POA (their Algorithm 1, stage 1).

MADII fills its replay buffer with transitions produced by these algorithms, so the
quality of the learner's warm start is set by them. FCM is in policies.py. HBA (Honey
Badger Algorithm) and POA (Pelican Optimization Algorithm) are population metaheuristics
that pick WHICH nodes are cluster heads, scored by the paper's Eq. (24) fitness with
a = 0.54 (their Table V):

    F1 = a * (mean residual energy of CHs / mean residual energy of non-CHs)
       + (1 - a) * (sum d(non-CH, sink) / (sum d(CH, sink) + sum d(non-CH, its CH)))

Higher is better: it rewards putting the head duty on nodes that still have energy, and
rewards clusterings whose total transport distance is short. K = 10 (Table I).
Members send to their nearest head; heads send to the sink.
"""
import numpy as np


def fitness(env, heads, alive):
    """Eq. (24). `heads` is an index array into the alive set."""
    n = env.n
    heads = np.asarray(heads)
    nch = np.setdiff1d(alive, heads)
    if heads.size == 0 or nch.size == 0:
        return -np.inf
    e_ch, e_nch = env.E[heads].mean(), env.E[nch].mean()
    d_nch_sink = env.d2s[nch].sum()
    d_ch_sink = env.d2s[heads].sum()
    d_nch_ch = env.d[np.ix_(nch, heads)].min(axis=1).sum()
    a = 0.54
    return a * (e_ch / max(e_nch, 1e-12)) + (1 - a) * (d_nch_sink / max(d_ch_sink + d_nch_ch, 1e-12))


def _assign(env, heads, alive):
    parent = np.full(env.n, env.n, int)
    nch = np.setdiff1d(alive, heads)
    if nch.size:
        parent[nch] = heads[env.d[np.ix_(nch, heads)].argmin(axis=1)]
    parent[heads] = env.n
    return parent


def _search(env, rng, k, iters, pop, update):
    """Shared loop for the two metaheuristics: a population of CH sets, refined by
    `update`, scored by Eq. (24). Table V: 800 iterations; we use far fewer per round
    because the routing decision is remade every round (documented deviation)."""
    alive = np.flatnonzero(env.alive)
    k = int(min(k, alive.size))
    if k == 0:
        return np.full(env.n, env.n, int)
    P = [rng.choice(alive, k, replace=False) for _ in range(pop)]
    F = [fitness(env, p, alive) for p in P]
    best = int(np.argmax(F))
    for t in range(iters):
        for i in range(pop):
            cand = update(P[i], P[best], alive, rng, t, iters)
            f = fitness(env, cand, alive)
            if f > F[i]:
                P[i], F[i] = cand, f
                if f > F[best]:
                    best = i
    return _assign(env, P[best], alive)


def _mutate(sol, guide, alive, rng, frac):
    """Replace a fraction of the current heads, biased toward the incumbent best."""
    sol = np.array(sol)
    n_swap = max(1, int(frac * sol.size))
    pos = rng.choice(sol.size, n_swap, replace=False)
    pool = np.setdiff1d(alive, sol)
    if pool.size == 0:
        return sol
    take_guide = rng.random(n_swap) < 0.5
    picks = rng.choice(pool, n_swap, replace=True)
    gpicks = rng.choice(guide, n_swap, replace=True)
    sol[pos] = np.where(take_guide, gpicks, picks)
    return np.unique(sol) if np.unique(sol).size == sol.size else sol


def hba_routing(env, k=10, iters=15, pop=10, rng=None):
    """Honey Badger Algorithm: density factor shrinks the search over iterations
    (digging -> honey phase), which we model as a shrinking mutation fraction."""
    rng = rng or np.random.default_rng(0)

    def update(sol, best, alive, r, t, T):
        density = 2.0 * np.exp(-t / max(T, 1))        # HBA's alpha, decaying
        return _mutate(sol, best, alive, r, min(0.9, 0.15 * density))

    return _search(env, rng, k, iters, pop, update)


def poa_routing(env, k=10, iters=15, pop=10, rng=None):
    """Pelican Optimization: an exploration phase toward a random prey, then an
    exploitation phase of shrinking local moves."""
    rng = rng or np.random.default_rng(0)

    def update(sol, best, alive, r, t, T):
        if t < T / 2:                                   # phase 1: move toward prey
            prey = r.choice(alive, min(sol.size, alive.size), replace=False)
            return _mutate(sol, prey, alive, r, 0.4)
        return _mutate(sol, best, alive, r, 0.1)        # phase 2: local search

    return _search(env, rng, k, iters, pop, update)
