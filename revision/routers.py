"""Routers for the revision bench. All compute the round's tree at the sink.

Proposed:   BookedLoadDijkstra   - predictive.LoadAwareDijkstra (fixed booking), rates observed
                                   by the sink from the packets it receives: no extra bits
Ablations:  LEST tiers           - predictive.LESTDijkstra, load reported as 4 tiers (charged)
            load only            - booking without the battery weight
Baselines (same family: multi-hop trees, no aggregation):
            min-energy Dijkstra, battery-weighted Dijkstra
            FlowAugmentation     - Chang & Tassiulas (2004) flow augmentation run as a per-round
                                   planner: solver-free, residual-energy link costs
            LoadBalancedSPT      - Shan et al. (2013) top-down load-balanced shortest-path tree
            LP planner           - predictive.ForecastLP v2 (upper reference, needs an LP solver)
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, maximum_flow

import dijkstra_rl as D
from predictive import ForecastLP, LESTDijkstra, LoadAwareDijkstra

E_RX_BIT = 50e-9


class _ObservedRates:
    """Mean packets per round of each node over the last `window` rounds, as the sink counts
    them from received packets (A11); initialised from the pre-deployment history."""

    def __init__(self, history, window=24):
        self.window = window
        self.recent = [np.asarray(x, float) for x in history[-window:]]

    def __call__(self, w):
        if w.round > 0 and w.traffic is not None:
            self.recent = (self.recent + [np.asarray(w.traffic[w.round - 1], float)])[-self.window:]
        if not self.recent:
            return np.ones(w.n)
        return np.mean(self.recent, axis=0)


def _subtree_bits(parent, own, n):
    """Bits each node transmits under tree `parent` when node i originates own[i] bits."""
    depth = np.zeros(n, int)
    for i in range(n):
        u, k = i, 0
        while u != n and k <= n:
            u = parent[u]; k += 1
        depth[i] = k
    load = own.astype(float).copy()
    for i in np.argsort(-depth):
        if parent[i] != n:
            load[parent[i]] += load[i]
    return load


class FlowAugmentation:
    """Chang & Tassiulas flow augmentation as a per-round planner.

    Over a planning horizon of H rounds, every source's traffic is pushed to the sink in K
    increments; before each increment the shortest-cost paths are recomputed with the link cost
        c(i, j) = e_tx(i, j) (E0 / E'_i)^x + e_rx (E0 / E'_j)^x
    where E' is the residual energy predicted after the increments already placed (energy at
    both ends of the link, as in [Chang & Tassiulas 2004]). The round's tree sends each node to
    the next hop that carries the largest share of its augmented flow. No solver is used."""

    def __init__(self, w, history, x=2.0, H=24, K=10, window=24):
        self.x, self.H, self.K = x, H, K
        self.rates = _ObservedRates(history, window)

    def __call__(self, w):
        n = w.n
        rate = self.rates(w)
        alive = w.alive
        Ep = np.where(alive, w.E, 0.0).astype(float)
        delta = np.where(alive, rate * w.L, 0.0) * self.H / self.K
        ctrl = self.H / self.K * E_RX_BIT * w.LC
        F = np.zeros((n, n + 1))
        base_tx = w.etx_bit[:n]
        for _ in range(self.K):
            m = (w.e0 / np.maximum(Ep, 1e-6 * w.e0)) ** self.x
            G = np.zeros((n + 1, n + 1))
            G[:n] = base_tx * m[:, None]
            G[:n, :n] += E_RX_BIT * m[None, :]
            np.fill_diagonal(G, 0.0)
            dead = np.flatnonzero(~alive)
            G[dead, :] = 0.0; G[:, dead] = 0.0
            _, pred = dijkstra(G.T, directed=True, indices=n, return_predecessors=True)
            p = pred[:n].astype(int); p[p < 0] = n; p[~alive] = n
            load = _subtree_bits(p, delta, n)
            F[np.arange(n), p] += np.where(alive, load, 0.0)
            rx = np.where(alive, load - delta, 0.0)
            Ep = Ep - np.where(alive, load * base_tx[np.arange(n), p] + rx * E_RX_BIT + ctrl, 0.0)
        parent = np.argmax(F, axis=1)
        parent[~alive] = n
        return parent


class LoadBalancedSPT:
    """Shan et al. (2013) top-down construction of a load-balanced shortest-path tree.

    Hop layers are built over the radio range R. Layer l+1 is attached to layer l so that the
    maximum load of the sink's branches (children of the sink) is minimised: a bisection on
    the bound B with a max-flow feasibility test (unit load per sensor, as in their model).
    Within the chosen branch a node takes the candidate parent with the fewest descendants.
    The tree is rebuilt only when a node dies. Nodes with no neighbour in range of the
    previous layer send directly to the sink."""

    def __init__(self, w, history, R=125.0):
        self.R, self.parent, self.alive_key = R, None, None

    def __call__(self, w):
        key = w.alive.tobytes()
        if self.parent is None or key != self.alive_key:
            self.parent, self.alive_key = self._build(w), key
        return self.parent

    def _build(self, w):
        n, R = w.n, self.R
        alive = np.flatnonzero(w.alive)
        parent = np.full(n, n)
        layer_of = np.full(n + 1, -1); layer_of[n] = 0
        branch = np.full(n, -1)
        desc = np.zeros(n)
        frontier = [n]
        unassigned = set(alive.tolist())
        blen = {}
        l = 0
        while unassigned:
            nxt = [v for v in unassigned if (w.d[v, frontier] <= R).any()]
            if not nxt:                       # out of range of every node of the last layer
                for v in unassigned:
                    parent[v] = n; branch[v] = v; blen[v] = 1
                break
            if frontier == [n]:              # layer 1: each child of the sink is its own branch
                for v in nxt:
                    parent[v] = n; branch[v] = v; blen[v] = 1
            else:
                self._attach(w, nxt, frontier, parent, branch, blen, desc)
            for v in nxt:
                layer_of[v] = l + 1
                unassigned.discard(v)
            frontier, l = nxt, l + 1
        return parent

    def _attach(self, w, nodes, prev, parent, branch, blen, desc):
        R = self.R
        cand = {v: [u for u in prev if w.d[v, u] <= R] for v in nodes}
        branches = sorted({branch[u] for u in prev})
        bidx = {b: k for k, b in enumerate(branches)}
        nv, nb = len(nodes), len(branches)
        cur = np.array([blen[b] for b in branches])

        def assign(B):
            # nodes: source 0, node k -> 1 + k, branch b -> 1 + nv + b, sink -> 1 + nv + nb
            rows, cols, cap = [], [], []
            for k, v in enumerate(nodes):
                rows.append(0); cols.append(1 + k); cap.append(1)
                for b in {bidx[branch[u]] for u in cand[v]}:
                    rows.append(1 + k); cols.append(1 + nv + b); cap.append(1)
            for b in range(nb):
                rows.append(1 + nv + b); cols.append(1 + nv + nb); cap.append(int(max(B - cur[b], 0)))
            N = nv + nb + 2
            G = csr_matrix((np.array(cap, np.int32), (rows, cols)), shape=(N, N))
            res = maximum_flow(G, 0, N - 1)
            if res.flow_value < nv:
                return None
            flow = res.flow.tocoo()
            out = {}
            for r, c, f in zip(flow.row, flow.col, flow.data):
                if f > 0 and 1 <= r <= nv and c > nv:
                    out[nodes[r - 1]] = branches[c - 1 - nv]
            return out

        lo, hi = int(cur.max()), int(cur.max()) + nv
        best = assign(hi)
        while lo < hi:
            mid = (lo + hi) // 2
            got = assign(mid)
            if got is None:
                lo = mid + 1
            else:
                hi, best = mid, got
        for v in nodes:
            b = best[v]
            opts = [u for u in cand[v] if branch[u] == b]
            u = min(opts, key=lambda u: (desc[u], w.d[v, u]))
            parent[v] = u; branch[v] = b; blen[b] += 1
            x = u
            while x != w.n:
                desc[x] += 1; x = parent[x]


def build(name, w, history, params=None):
    """Router by name. `params` overrides the tuned settings (used by the tuning scripts)."""
    p = dict(params or {})
    if name == "Min-energy Dijkstra":
        return lambda ww: D.dijkstra_tree(ww, np.ones(ww.n))
    if name == "Battery-weighted Dijkstra":
        return D.fast_battery_weighted
    if name == "Booked-load Dijkstra":
        return LoadAwareDijkstra(w, history, kappa=p.get("kappa", TUNED["bld_kappa"]), order="near")
    if name == "Booked-load, LEST 4-tier load (ablation)":
        return LESTDijkstra(w, history, kappa=p.get("kappa", TUNED["bld_kappa"]), levels="lest", rule="original", band=0.05, exact_energy=True)
    if name == "Booked-load, load only (ablation)":
        return LoadAwareDijkstra(w, history, kappa=p.get("kappa", TUNED["bld_kappa"]), order="near", energy=False)
    if name == "Flow augmentation (Chang-Tassiulas)":
        return FlowAugmentation(w, history, x=p.get("x", TUNED["fa_x"]), H=p.get("H", TUNED["fa_H"]), K=p.get("K", TUNED["fa_K"]))
    if name == "Load-balanced SPT (Shan et al.)":
        return LoadBalancedSPT(w, history, R=p.get("R", TUNED["shan_R"]))
    if name == "LP planner (every round)":
        return ForecastLP(w, history, forecaster="history-mean", resolve_every=1, mu=1e-3)
    raise KeyError(name)


# Settings chosen on validation deployments 500-511 (revision/tune.py); see results/revision/tuning.json
TUNED = {"fa_x": 1.0, "fa_H": 3, "fa_K": 30, "shan_R": 110.0, "bld_kappa": 0.5}

MAIN = ["Min-energy Dijkstra", "Battery-weighted Dijkstra", "Load-balanced SPT (Shan et al.)",
        "Flow augmentation (Chang-Tassiulas)", "Booked-load Dijkstra",
        "Booked-load, LEST 4-tier load (ablation)", "Booked-load, load only (ablation)",
        "LP planner (every round)"]
