"""Classical routing policies + the FCM clustering baseline MADII compares against.

Every policy is a callable: policy(env) -> parent array of length n (values 0..n, n = sink).
"""
import heapq
import numpy as np

from env import etx, erx


def direct(env):
    return np.full(env.n, env.n, int)


def _dijkstra(env, weight):
    """Shortest path to the sink over alive nodes. weight(i, j) is the cost of i -> j."""
    n = env.n
    alive = np.flatnonzero(env.alive)
    dist = {n: 0.0}
    parent = np.full(n, n, int)
    pq = [(0.0, n)]
    done = set()
    while pq:
        dj, j = heapq.heappop(pq)
        if j in done:
            continue
        done.add(j)
        for i in alive:
            if i in done:
                continue
            w = dj + weight(i, j)
            if w < dist.get(i, np.inf):
                dist[i] = w
                parent[i] = j
                heapq.heappush(pq, (w, i))
    return parent


def min_energy(env):
    """Least total joules per round: per-bit TX at i plus RX at the relay.
    In this model that is the exact per-round energy optimum (costs are linear in bits)."""
    n = env.n
    return _dijkstra(env, lambda i, j: env.etx_bit[i, j] + (0.0 if j == n else 50e-9))


def battery_weighted(env, alpha=1.0):
    """Our `coordinator`: same cost scaled by (E0 / residual of the SENDER)^alpha, so
    paths route around drained relays. One line different from min_energy."""
    n = env.n
    E = env.E

    def w(i, j):
        base = env.etx_bit[i, j] + (0.0 if j == n else 50e-9)
        return base * (env.e0 / max(E[i], 1e-12)) ** alpha

    return _dijkstra(env, w)


def fcm_routing(env, k=10, m=2.0, iters=30, rng=None, ch_rule="energy"):
    """Fuzzy C-means clustering (their Table I: clustering number 10).
    Cluster head = highest residual energy in the cluster; members send to their head,
    heads send straight to the sink. This is the FCM baseline row of their Table VI."""
    n = env.n
    alive = np.flatnonzero(env.alive)
    if alive.size == 0:
        return np.full(n, n, int)
    k = int(min(k, alive.size))
    pos = env.pos[alive]
    rng = rng or np.random.default_rng(0)
    c = pos[rng.choice(alive.size, k, replace=False)].astype(float)
    for _ in range(iters):
        d = np.linalg.norm(pos[:, None, :] - c[None, :, :], axis=-1).clip(1e-9)
        u = 1.0 / (d ** (2 / (m - 1)))
        u /= u.sum(1, keepdims=True)
        w = u ** m
        c = (w.T @ pos) / w.sum(0)[:, None]
    lab = np.argmin(np.linalg.norm(pos[:, None, :] - c[None, :, :], axis=-1), axis=1)
    parent = np.full(n, n, int)
    for cl in range(k):
        members = alive[lab == cl]
        if members.size == 0:
            continue
        if ch_rule == "energy":                 # most residual energy in the cluster
            head = members[np.argmax(env.E[members])]
        elif ch_rule == "near_sink":             # closest to the sink (cheapest uplink)
            head = members[np.argmin(env.d2s[members])]
        else:                                    # closest to the fuzzy centroid
            head = members[np.argmin(np.linalg.norm(env.pos[members] - c[cl], axis=1))]
        parent[members] = head
        parent[head] = n
    return parent


def random_routing(env, rng=None):
    """Uniform random next hop among alive nodes and the sink (sanity floor)."""
    rng = rng or np.random.default_rng(0)
    n = env.n
    alive = np.flatnonzero(env.alive)
    choices = np.append(alive, n)
    return rng.choice(choices, size=n)
