"""Warm-started RL Dijkstra: reinforcement learning on top of battery-weighted Dijkstra.

The router is still Dijkstra. What is learned is a per-node correction to its edge cost:

    w(i, j) = [etx_bit(i, j) + erx_bit] * (E0 / E_i)^alpha * exp(m_theta(phi_i))

m_theta is a small MLP shared by every node, read from each node's state phi_i every
round. Its output layer starts at ZERO, so the untrained policy IS battery-weighted
Dijkstra (`policies.battery_weighted`): the learner starts from the classical answer
(88% of the provable optimum), not from scratch, and can only be kept if it improves.

State phi_i (per node, per round), computed from the warm-start tree of this round:
    bias, E_i/E0, log(E0/E_i), d(i, sink)/diag, subtree share (bits relayed / all bits),
    lifetime pressure T_min/T_i, log(T_i / mean T)
where T_i = E_i / (joules i would spend this round) = projected rounds-to-death. The
LP optimum equalises exactly this quantity, so it is the natural signal to learn from.

Training: episodic policy search with parameter-space exploration (PGPE, Sehnke et al.
2010; the antithetic estimator of Salimans et al. 2017). Each antithetic pair is run on
the same training deployment, so the deployment's difficulty cancels in the difference.
Return = rounds to first node death / LP ceiling T* of that deployment (the ceiling
normalises away easy vs hard deployments). Validation seeds pick the checkpoint; the
warm start itself is the candidate to beat, so the output is never worse than Dijkstra
on validation.

Seeds: train >= 1000, validation 500-519, evaluation 0-29 (same as evaluate.py).

  .venv/bin/python dijkstra_rl.py --iters 60            # train, writes checkpoints/dijkstra_rl.npz
"""
import argparse, json, math, time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

import env as E
from lp_bound import max_lifetime_T

N_IN, DIAG = 7, math.hypot(500.0, 500.0)


def dijkstra_tree(w, node_mult):
    """Shortest-path tree to the sink with edge cost base(i,j) * node_mult[i].
    Vectorised (scipy); identical output to policies._dijkstra for the same weights."""
    n = w.n
    G = np.zeros((n + 1, n + 1))
    C = w.etx_bit[:n].copy()
    C[:, :n] += 50e-9                                      # relay's receive cost
    G[:n] = C * node_mult[:, None]
    np.fill_diagonal(G, 0.0)
    dead = np.flatnonzero(~w.alive)
    G[dead, :] = 0.0; G[:, dead] = 0.0                     # 0 = no edge in csgraph
    _, pred = dijkstra(G.T, directed=True, indices=n, return_predecessors=True)
    p = pred[:n].astype(int)
    p[p < 0] = n
    return p


def dijkstra_tree_rx(w, tx_mult, rx_mult):
    """As dijkstra_tree, but each link's receive energy is weighted by the RECEIVER's
    multiplier, as the max-lifetime LP charges it, instead of the sender's."""
    n = w.n
    G = np.zeros((n + 1, n + 1))
    G[:n] = w.etx_bit[:n] * tx_mult[:, None]
    G[:n, :n] += 50e-9 * rx_mult[None, :]
    np.fill_diagonal(G, 0.0)
    dead = np.flatnonzero(~w.alive)
    G[dead, :] = 0.0; G[:, dead] = 0.0
    _, pred = dijkstra(G.T, directed=True, indices=n, return_predecessors=True)
    p = pred[:n].astype(int)
    p[p < 0] = n
    return p


def battery_mult(w, alpha=1.0):
    return (w.e0 / np.maximum(w.E, 1e-12)) ** alpha


def fast_battery_weighted(w, alpha=1.0):
    return dijkstra_tree(w, battery_mult(w, alpha))


def node_features(w, p):
    """phi (n, N_IN) from the tree p this round would use (see module docstring)."""
    n = w.n
    p = w._sanitize(p)
    load = w._loads(p)                                     # bits each node transmits
    rx = np.where(w.alive, load - w.L, 0.0).clip(min=0)
    spend = load * w.etx_bit[np.arange(n), p] + E.erx(rx) + E.erx(w.LC)
    T = np.where(w.alive, w.E / np.maximum(spend, 1e-15), np.inf)
    a = w.alive
    Tmin, Tmean = T[a].min(), T[a].mean()
    f = np.zeros((n, N_IN))
    f[:, 0] = 1.0
    f[:, 1] = w.E / w.e0
    f[:, 2] = np.log(w.e0 / np.maximum(w.E, 1e-6 * w.e0)).clip(max=5.0)
    f[:, 3] = w.d2s / DIAG
    f[:, 4] = load / (w.L * max(a.sum(), 1))
    f[:, 5] = np.where(a, Tmin / T, 0.0)
    f[:, 6] = np.where(a, np.log(np.maximum(T, 1e-9) / Tmean), 0.0).clip(-3, 3)
    return f


class Policy:
    """theta = flat vector [W1 (N_IN*H), b1 (H), W2 (H), b2 (1)]; zero W2/b2 = warm start."""

    def __init__(self, hidden=16, theta=None, alpha=1.0, seed=0):
        self.h, self.alpha = hidden, alpha
        self.dim = N_IN * hidden + hidden + hidden + 1
        if theta is None:
            rng = np.random.default_rng(seed)
            theta = np.zeros(self.dim)
            theta[:N_IN * hidden] = rng.normal(0, 1 / math.sqrt(N_IN), N_IN * hidden)
        self.theta = np.asarray(theta, float)

    def mult(self, phi):
        h, t = self.h, self.theta
        W1 = t[:N_IN * h].reshape(N_IN, h)
        b1 = t[N_IN * h:N_IN * h + h]
        W2 = t[N_IN * h + h:N_IN * h + 2 * h]
        b2 = t[-1]
        m = np.tanh(phi @ W1 + b1) @ W2 + b2
        return np.exp(m.clip(-3.0, 3.0))

    def __call__(self, w):
        base = battery_mult(w, self.alpha)
        p0 = dijkstra_tree(w, base)                         # warm-start decision
        return dijkstra_tree(w, base * self.mult(node_features(w, p0)))


@lru_cache(maxsize=None)
def ceiling(seed, n=100):
    return max_lifetime_T(E.WSN(n=n, seed=seed))


def fnd_ratio(args):
    """Rounds to first node death / LP ceiling, for one (theta, seed)."""
    theta, seed, hidden, n = args
    w = E.WSN(n=n, seed=seed)
    pol = Policy(hidden, theta)
    w.reset()
    for _ in range(1000):
        info = w.step(pol(w))
        if info["n_dead"] >= 1:
            break
    return w.round / ceiling(seed, n)


def train(args):
    rng = np.random.default_rng(args.seed)
    pol = Policy(args.hidden, seed=args.seed)
    theta, dim = pol.theta.copy(), pol.dim
    m_adam, v_adam = np.zeros(dim), np.zeros(dim)
    val_seeds = list(range(500, 500 + args.val_seeds))
    Path("checkpoints").mkdir(exist_ok=True); Path("logs").mkdir(exist_ok=True)
    log = open(f"logs/{args.tag}.jsonl", "a")
    pool = Pool(args.workers)

    def val(th):
        return float(np.mean(pool.map(fnd_ratio, [(th, s, args.hidden, args.nodes)
                                                  for s in val_seeds])))

    best_val = val(theta)                                   # the warm start = Dijkstra
    best_theta, t0 = theta.copy(), time.time()
    rec = {"iter": 0, "val_fnd_over_ceiling": round(best_val, 4), "note": "warm start"}
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n")

    for it in range(1, args.iters + 1):
        eps = rng.normal(size=(args.pairs, dim))
        seeds = rng.integers(1000, 100000, size=args.pairs)
        jobs = []
        for k in range(args.pairs):
            jobs.append((theta + args.sigma * eps[k], int(seeds[k]), args.hidden, args.nodes))
            jobs.append((theta - args.sigma * eps[k], int(seeds[k]), args.hidden, args.nodes))
        F = np.array(pool.map(fnd_ratio, jobs)).reshape(args.pairs, 2)
        diff = F[:, 0] - F[:, 1]
        scale = diff.std() + 1e-8
        g = (diff / scale) @ eps / (2 * args.pairs)         # ascent direction
        m_adam = 0.9 * m_adam + 0.1 * g
        v_adam = 0.999 * v_adam + 0.001 * g * g
        step = args.lr * (m_adam / (1 - 0.9 ** it)) / (np.sqrt(v_adam / (1 - 0.999 ** it)) + 1e-8)
        theta = theta + step
        rec = {"iter": it, "train_mean": round(float(F.mean()), 4),
               "mins": round((time.time() - t0) / 60, 1)}
        if it % args.val_every == 0 or it == args.iters:
            v = val(theta)
            rec["val_fnd_over_ceiling"] = round(v, 4)
            if v > best_val:
                best_val, best_theta = v, theta.copy()
                rec["best"] = True
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()

    np.savez(f"checkpoints/{args.tag}.npz", theta=best_theta, hidden=args.hidden,
             val=best_val, args=json.dumps(vars(args)))
    print(f"saved checkpoints/{args.tag}.npz  best validation FND/T* = {best_val:.4f}")


def load(path="checkpoints/dijkstra_rl.npz"):
    z = np.load(path)
    return Policy(int(z["hidden"]), z["theta"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dijkstra_rl")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--val-seeds", type=int, default=20)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())
