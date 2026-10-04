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

    mu = 0.0                                              # min-energy tie-break (lp_bound)

    def _build(self, w):
        gen = np.maximum(self._rates(w), 0.05) * w.L
        self.T_plan, flow = max_lifetime_T(w, return_flow=True, gen=gen, mu=self.mu)
        old_t, old_c = getattr(self, "tgt", None), getattr(self, "credit", None)
        carry = getattr(self, "carry", False) and old_t
        if carry:                                         # book last round with the OLD split
            self._book_last_round(w, old_t, old_c)
        self.tgt = {i: np.array([j for j, _ in e]) for i, e in flow.items()}
        self.w = {i: np.array([g for _, g in e]) for i, e in flow.items()}
        self.credit = {i: np.zeros(len(e)) for i, e in flow.items()}
        if carry:                                         # keep each hop's rounding residual
            for i in self.tgt:
                if i in old_t:
                    m = dict(zip(old_t[i].tolist(), old_c[i].tolist()))
                    self.credit[i] = np.array([m.get(j, 0.0) for j in self.tgt[i].tolist()])
        self.prev_k = {}
        self.n_solves = getattr(self, "n_solves", 0) + 1


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
    long-run mean; the forecast gives the LP a stable, forward-looking rate.

    Planner options (defaults = the published Part 3 configuration):
      split  "rounds": smooth weighted round-robin over rounds (each round a node sends
             everything to one next hop, hops rotate in proportion to the LP split);
             "bits": deficit round-robin over the bits actually sent, so a surge round
             with 4x traffic counts 4x towards its hop's share (causal: it books last
             round's bits, which the node knows once it has sent them).
      mu     min-energy tie-break among max-lifetime flows (lp_bound.max_lifetime_T).
      H      forecast horizon in rounds, or "life": the remaining lifetime the last
             LP solve predicted (at least one day), i.e. plan for the rate until death.
      event  re-solve early when a node's traffic last round exceeds its one-step
             forecast by more than `event` running mean absolute errors (a burst the
             forecast did not see); None = never.
      carry  keep each next hop's round-robin credit across re-solves instead of resetting
             it, so the rounding residual of one block is paid back in the next (with a
             reset every 4 rounds a 0.6/0.4 split is realised as 0.5/0.5 in every block).
    """
    def __init__(self, w, history, H=24, forecaster="holt-winters", resolve_every=4,
                 split="rounds", mu=0.0, event=None, carry=False, **fkw):
        self.H, self.split, self.mu, self.event, self.carry = H, split, mu, event, carry
        self.obs = _Observer(w.n, history, forecaster, **fkw)
        self.window, self.resolve_every, self.k, self.hist = 4, resolve_every, 0, []
        self.T_plan, self.prev, self.thresh = None, None, None
        self._build(w)

    def _horizon(self):
        if self.H == "life":
            return int(np.clip(round(self.T_plan or 24), 24, 400))
        return self.H

    def _rates(self, w):
        self.obs.catch_up(w)
        return self.obs.f.forecast(self._horizon())[0].mean(0)

    def _surprised(self, w):
        """True if last round's traffic broke the band forecast before it happened."""
        hit = self.thresh is not None and w.round > 0 and \
            bool(np.any(w.alive & (w.traffic[w.round - 1] > self.thresh)))
        self.obs.catch_up(w)                              # now forecast the coming round
        f = self.obs.f
        self.thresh = f.forecast(1)[0][0] + self.event * f.mae
        return hit

    def _book_last_round(self, w, tgt, credit):
        """Bits split only: book last round's bits into the credits before they are carried."""
        if self.split != "bits" or self.prev is None or not self.prev_k or w.round == 0:
            return
        own = np.where(w.alive, w.traffic[w.round - 1], 0).astype(float)
        sent = subtree_sum(w, self.prev, own)
        for i, kk in self.prev_k.items():
            credit[i] += self.w[i] * sent[i]
            credit[i][kk] -= sent[i]
        self.prev_k = {}

    def __call__(self, w):
        if self.split == "rounds" and self.event is None:
            return super().__call__(w)
        due = self.resolve_every and self.k % self.resolve_every == 0 and self.k > 0
        if (self.event is not None and self._surprised(w)) or due:
            self._build(w)
            self.k = 0
        self.k += 1
        return self._route(w) if self.split == "rounds" else self._route_bits(w)

    def _route_bits(self, w):
        n = w.n
        if self.prev is not None and w.round > 0:
            own = np.where(w.alive, w.traffic[w.round - 1], 0).astype(float)
            self.sent = subtree_sum(w, self.prev, own)   # packets each node sent last round
            for i, kk in self.prev_k.items():
                c = self.credit[i]
                c += self.w[i] * self.sent[i]
                c[kk] -= self.sent[i]
        guess = getattr(self, "sent", None)
        parent = np.full(n, n, int); self.prev_k = {}
        for i in np.flatnonzero(w.alive):
            if i not in self.tgt:
                continue
            t, c = self.tgt[i], self.credit[i]
            ok = (t == n) | w.alive[np.minimum(t, n - 1)]
            if not ok.any():
                continue
            # smooth deficit round-robin: credit as if this round's bits (guessed from last
            # round) had already arrived, so the largest share goes first; picking on the
            # bare deficit sends a 9% hop 1 round in 4 when credits reset every 4 rounds
            g = max(guess[i], 1.0) if guess is not None else 1.0
            k = int(np.argmax(np.where(ok, c + self.w[i] * g, -np.inf)))
            parent[i] = t[k]; self.prev_k[i] = k
        self.prev = parent
        return parent


class LifetimeDijkstra:
    """Predicted-lifetime Dijkstra: battery-weighted Dijkstra whose node weight is the
    forecast time to death instead of the battery alone. Each round (knowing traffic up to
    t-1), Holt-Winters forecasts every node's own rate F_i averaged over the next H rounds.

      mode "load": weight_i = (E0 / E_i) * (F_i / mean F)^k
                   a node forecast to be busy over the coming day stays expensive before,
                   during and after its surge; with equal forecasts this is exactly battery
                   Dijkstra.
      mode "tau":  weight_i = (E0 / E_i) * (median tau / tau_i)^k, tau_i = E_i / drain_i the
                   forecast rounds node i has left, where drain_i is
                   the energy per round node i would spend sending its own forecast load
                   plus everything forecast to flow through it on the tree it is
                   actually using, smoothed over rounds (rate rho) so the tree settles
                   rather than flipping onto whichever nodes looked idle last round.
    """
    def __init__(self, w, history, H=24, mode="load", k=1.0, rho=0.1, forecaster="holt-winters", **fkw):
        self.obs = _Observer(w.n, history, forecaster, **fkw)
        self.H, self.mode, self.k, self.rho = H, mode, k, rho
        self.p, self.drain = None, None

    def __call__(self, w):
        self.obs.catch_up(w)
        F = np.clip(self.obs.f.forecast(self.H)[0].mean(0), 1e-3, None)
        base = D.battery_mult(w)
        if self.mode == "load":
            return D.dijkstra_tree(w, base * (F / F[w.alive].mean()) ** self.k)
        p0 = w._sanitize(self.p if self.p is not None else D.dijkstra_tree(w, base))
        through = subtree_sum(w, p0, F) * w.L
        drain = through * w.etx_bit[np.arange(w.n), p0] + E.erx(through - F * w.L)
        self.drain = drain if self.drain is None else (1 - self.rho) * self.drain + self.rho * drain
        tau = np.maximum(w.E, 1e-12) / np.maximum(self.drain, 1e-18)       # forecast rounds left
        self.p = D.dijkstra_tree(w, base * (np.median(tau[w.alive]) / tau) ** self.k)
        return self.p


class PriceDijkstra:
    """Price-guided Dijkstra: battery-weighted Dijkstra every round, steered by the
    max-lifetime LP's energy prices, re-solved only every `every` rounds (default once a day).

    LP duality: price[i] (the dual of node i's energy budget) is how many rounds of network
    lifetime one more joule at node i would buy, so it is high for the bottleneck nodes near
    the sink and zero for nodes with energy to spare. Each round:
        weight_i = (E0 / E_i) * (1 + c * price_i / mean positive price)
    The battery factor keeps Dijkstra's round-by-round balancing; the price factor adds the
    whole-network view the greedy tree lacks. rx="receiver" weights each link's receive energy
    by the receiver's weight, as the LP does (battery Dijkstra charges it to the sender).
    eta > 0 switches to dual ascent: weight_i = price_i * (E0/E_i)^c, with the LP price
    re-synced every `every` rounds and, in between, multiplied each round by
    exp(eta * (spent_i / share_i - 1)), share_i = E_i / (planned rounds left).
    Rates for the LP come from the forecaster
    (Holt-Winters by default) over the next H rounds. c = 0 is exactly battery Dijkstra.
    """
    def __init__(self, w, history, c=1.0, every=24, H=24, rx="sender", eta=0.0, forecaster="holt-winters", **fkw):
        self.obs = _Observer(w.n, history, forecaster, **fkw)
        self.c, self.every, self.H, self.k, self.rx, self.eta = c, every, H, 0, rx, eta
        self.n_solves = 0
        self.price = np.zeros(w.n)
        self.E_prev, self.T_end = None, None

    def _solve(self, w):
        self.obs.catch_up(w)
        gen = np.maximum(self.obs.f.forecast(self.H)[0].mean(0), 0.05) * w.L
        T, lam = max_lifetime_T(w, gen=gen, return_price=True)
        pos = lam[lam > 0]
        self.price = lam / pos.mean() if pos.size else lam
        self.T_end = w.round + T
        self.n_solves += 1

    def __call__(self, w):
        if (self.c or self.eta) and self.k % self.every == 0:
            self._solve(w)
        elif self.eta and self.E_prev is not None:
            # dual ascent between solves: a node that spent more than its share of the
            # remaining planned lifetime last round gets dearer, one that spent less cheaper
            spent = np.maximum(self.E_prev - w.E, 0.0)
            share = w.E / max(self.T_end - w.round, 1.0)
            ratio = np.where(w.alive & (share > 0), spent / np.maximum(share, 1e-15), 1.0)
            self.price = self.price * np.exp(self.eta * np.clip(ratio - 1.0, -1.0, 3.0))
        self.E_prev = w.E.copy()
        self.k += 1
        if self.eta:                                   # pure price routing, battery inside the price
            m = np.maximum(self.price, 1e-6) * D.battery_mult(w) ** self.c
        else:
            m = D.battery_mult(w) * (1.0 + self.c * self.price)
        return D.dijkstra_tree(w, m) if self.rx == "sender" else D.dijkstra_tree_rx(w, m, m)
