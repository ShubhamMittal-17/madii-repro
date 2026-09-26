"""Classical LP-flow routing, ported from the Phase-1 bench (phase1/lp_flow.py) to this one.

LPFlowRouting: solve the max-lifetime LP once, then realise its fractional flow with a
smooth weighted round-robin: each round every node picks ONE next hop, rotating among its
LP targets in proportion to the optimal split. Time-averaged, the routing equals the LP
flow. No learning.

LPFlowAdaptive: the same, but re-solve the LP every `resolve_every` rounds from the
current residual energies and alive set, which corrects the drift of the static version.

Both are stateful: build one per episode, e.g. E.run_episode(w, LPFlowAdaptive(w)).
"""
import numpy as np

import env as E
from lp_bound import max_lifetime_T


class LPFlowRouting:
    resolve_every = None

    def __init__(self, env, resolve_every=None):
        self.resolve_every = resolve_every or self.resolve_every
        self.k = 0
        self._build(env)

    def _build(self, env):
        _, flow = max_lifetime_T(env, return_flow=True)
        self.tgt = {i: np.array([j for j, _ in e]) for i, e in flow.items()}
        self.w = {i: np.array([g for _, g in e]) for i, e in flow.items()}
        self.credit = {i: np.zeros(len(e)) for i, e in flow.items()}

    def __call__(self, env):
        if self.resolve_every and self.k % self.resolve_every == 0 and self.k > 0:
            self._build(env)
        self.k += 1
        n = env.n
        parent = np.full(n, n, int)
        for i in np.flatnonzero(env.alive):
            if i not in self.tgt:
                continue
            t, c = self.tgt[i], self.credit[i]
            c += self.w[i]                                      # smooth WRR
            ok = (t == n) | env.alive[np.minimum(t, n - 1)]
            if not ok.any():
                continue
            k = int(np.argmax(np.where(ok, c, -np.inf)))
            c[k] -= 1.0
            parent[i] = t[k]
        return parent


class LPFlowAdaptive(LPFlowRouting):
    resolve_every = 8


if __name__ == "__main__":
    from lp_bound import max_lifetime_T as T
    import dijkstra_rl as D
    for s in range(3):
        w = E.WSN(seed=s)
        t = T(w)
        row = [f"seed {s}: T* {t:.1f}"]
        for name, pol in [("battery", D.fast_battery_weighted), ("static", LPFlowRouting(w)),
                          ("adaptive", LPFlowAdaptive(w))]:
            fnd = E.run_episode(E.WSN(seed=s), pol, (0.01,))[0.01]["SR"]
            row.append(f"{name} {fnd} ({100 * (fnd - 1) / t:.0f}%)")
        print("  ".join(row), flush=True)
