"""Part 3 routers for time-varying traffic: forecast-driven LP routing (ours), predictive
battery-weighted Dijkstra (an ablation), and the reactive and static LP baselines.

PredictiveDijkstra, each round t (it only knows traffic up to round t-1):
  1. ordinary battery-weighted tree p0 for the current batteries
  2. forecast each node's own traffic for the next H rounds, and keep only the EXTRA load
     beyond its current rate (battery Dijkstra already reacts to the current rate)
  3. reserve R_i = energy node i would spend over those H rounds on that extra load, its
     own plus everything forecast to flow through it on p0
  4. route on effective energy  E_eff = max(E - lam * R, eps * E)
lam = 0 is exactly reactive battery-weighted Dijkstra (tested).

Policies are stateful: build one per episode with the deployment's traffic history.
"""
import numpy as np

import env as E
import dijkstra_rl as D
from forecast import FORECASTERS
from lp_bound import max_lifetime_T
from lp_flow import LPFlowRouting


def subtree_sum(w, parent, values):
    """values[i] summed over each node's subtree in the forest `parent` (itself included)."""
    p = w._sanitize(parent)
    out = np.where(w.alive, values, 0.0).astype(float)
    for i in np.argsort(-w._depth(p)):              # deepest first
        if w.alive[i] and p[i] != w.n:
            out[p[i]] += out[i]
    return out


class _Observer:
    """Feeds the forecaster each round's traffic once that round has happened."""
    def __init__(self, n, history, forecaster="holt-winters", **fkw):
        self.f = FORECASTERS[forecaster](n, **fkw)
        self.f.fit_history(history)
        self.seen = 0

    def catch_up(self, w):
        while self.seen < w.round:                  # rounds 0..w.round-1 are in the past
            self.f.update(w.traffic[self.seen]); self.seen += 1


class PredictiveDijkstra:
    def __init__(self, w, history, lam=1.0, H=12, forecaster="holt-winters", eps=0.05, window=4, **fkw):
        self.obs = _Observer(w.n, history, forecaster, **fkw)
        self.lam, self.H, self.eps, self.window = lam, H, eps, window
        self.recent = [np.asarray(x, float) for x in history[-window:]]

    def __call__(self, w):
        base = D.battery_mult(w)
        if self.lam == 0:
            return D.dijkstra_tree(w, base)
        self.obs.catch_up(w)
        self.recent = (self.recent + [w.traffic[t] for t in range(max(0, w.round - 1), w.round)])[-self.window:]
        now = np.mean(self.recent, axis=0)               # current rate: what battery Dijkstra already sees
        mean, _ = self.obs.f.forecast(self.H)
        extra = np.clip(mean.sum(0) - self.H * now, 0, None) * w.L   # load the forecast adds beyond today
        p0 = D.dijkstra_tree(w, base)
        through = subtree_sum(w, p0, extra)              # extra bits each node will transmit (own + relayed)
        p = w._sanitize(p0)
        R = through * w.etx_bit[np.arange(w.n), p] + E.erx(through - extra)
        E_eff = np.maximum(w.E - self.lam * R, self.eps * w.E)   # floor relative to own battery
        return D.dijkstra_tree(w, w.e0 / E_eff)


class ReactiveLP(LPFlowRouting):
    """Adaptive LP-flow that re-solves every k rounds from current batteries, using each
    node's recent traffic rate (mean of the last `window` rounds) as its generation."""
    def __init__(self, w, history, resolve_every=4, window=4):
        self.hist = [np.asarray(h, float) for h in history[-window:]]
        self.window, self.resolve_every, self.k = window, resolve_every, 0
        self._build(w)

    def _rates(self, w):
        recent = self.hist + [w.traffic[t] for t in range(max(0, w.round - self.window), w.round)]
        return np.mean(recent[-self.window:], axis=0)

    def _build(self, w):
        gen = np.maximum(self._rates(w), 0.05) * w.L
        _, flow = max_lifetime_T(w, return_flow=True, gen=gen)
        self.tgt = {i: np.array([j for j, _ in e]) for i, e in flow.items()}
        self.w = {i: np.array([g for _, g in e]) for i, e in flow.items()}
        self.credit = {i: np.zeros(len(e)) for i, e in flow.items()}

    def __call__(self, w):
        if self.k % self.resolve_every == 0 and self.k > 0:
            self._build(w)
        return super().__call__(w)


class StaticLP(ReactiveLP):
    """The LP solved once, at deployment, from the historical mean rate per node."""
    def __init__(self, w, history):
        self.hist = [np.asarray(history, float).mean(0)]
        self.window, self.resolve_every, self.k = 1, 10 ** 9, 0
        self._build(w)

    def _rates(self, w):
        return self.hist[0]


class ForecastLP(ReactiveLP):
    """Forecast-driven LP routing (ours): re-solve the max-lifetime LP every k rounds from
    current batteries, with each node's generation set to its FORECAST mean rate over the
    next H rounds. A reactive LP plans from noisy recent counts and a static LP from the
    long-run mean; the forecast gives the LP a stable, forward-looking rate."""
    def __init__(self, w, history, H=24, forecaster="holt-winters", resolve_every=4, **fkw):
        self.H = H
        self.obs = _Observer(w.n, history, forecaster, **fkw)
        self.window, self.resolve_every, self.k, self.hist = 4, resolve_every, 0, []
        self._build(w)

    def _rates(self, w):
        self.obs.catch_up(w)
        return self.obs.f.forecast(self.H)[0].mean(0)
