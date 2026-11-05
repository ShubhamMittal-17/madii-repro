"""Realism probe (review comment M7): lossy links with ARQ, optionally with receiver-side
contention that grows with the traffic the routing puts near each receiver.

Link model (expected-value, so lifetime stays deterministic):
  p_ij   per-attempt packet reception rate, logistic in distance with static log-normal
         shadowing:  p = 1 / (1 + exp((d_ij * exp(X_ij) - D50) / WIDTH)),  X_ij ~ N(0, SIGMA)
  ARQ    up to MAX_TRIES attempts per hop: expected attempts a = (1 - (1-p)^M) / p and hop
         success s = 1 - (1-p)^M. Sender pays a transmissions, receiver a receptions.
  contention (optional): p_ij is multiplied by exp(-I_j / CAPACITY), where I_j is the number of
         bits transmitted this round (attempts included) by other nodes within INTERFERENCE_R of
         receiver j. Spreading traffic over more relays changes I_j, which is exactly the effect
         the review asks about.
Routers plan with the loss-aware static per-bit cost e_tx(i,j) * a_ij / s_ij (energy per
delivered bit, an ETX-style metric), which also enters the oracle bound. Contention is not
known to any router. Packets lost after MAX_TRIES are counted as not delivered; relays forward
only what they received.
"""
import numpy as np

import env as E
from revision.model import Deployment

D50, WIDTH, SIGMA, MAX_TRIES = 120.0, 15.0, 0.1, 4
INTERFERENCE_R, CAPACITY = 100.0, 2.0e6


def _arq(p):
    p = np.clip(p, 1e-6, 1.0)
    a = (1 - (1 - p) ** MAX_TRIES) / p
    s = 1 - (1 - p) ** MAX_TRIES
    return a, s


class LossyDeployment(Deployment):
    def __init__(self, contention=False, **kw):
        super().__init__(**kw)
        self.contention = contention
        rng = np.random.default_rng(kw.get("seed", 0) + 424242)
        X = rng.normal(0, SIGMA, size=self.d.shape)
        X = np.triu(X, 1); X = X + X.T                                   # symmetric shadowing
        self.prr = 1.0 / (1.0 + np.exp((self.d * np.exp(X) - D50) / WIDTH))
        a, s = _arq(self.prr)
        self.radio_bit = E.etx(1.0, self.d)                             # true energy per attempt-bit
        self.etx_bit = self.radio_bit * a / np.maximum(s, 1e-3)         # routers' and oracle's cost

    def step(self, parent):
        n = self.n
        own = self.own_bits()
        p = self._sanitize(parent)
        start_alive = self.alive.copy()
        start_energy = self.E.copy()
        idx = np.flatnonzero(start_alive)
        depth = self._depth(p)
        order = idx[np.argsort(-depth[idx])]

        prr = self.prr[np.arange(n), p].copy()
        if self.contention:                                          # first pass: loss-free loads
            tx0 = self._loads(p) * _arq(prr)[0]
            near = self.d[:, :n] <= INTERFERENCE_R                      # (n+1) x n
            I = near @ np.where(start_alive, tx0, 0.0)                  # bits sent near each receiver
            I_recv = I[p] - tx0                                         # exclude the sender itself
            prr = prr * np.exp(-np.maximum(I_recv, 0.0) / CAPACITY)
        a, s = _arq(prr)

        out = np.zeros(n); inc = np.zeros(n + 1); rx_bits = np.zeros(n + 1)
        frac = np.zeros(n)                                              # delivered fraction per source
        for i in order:                                                 # leaves first
            out[i] = own[i] + inc[i]
            j = p[i]
            rx_bits[j] += out[i] * a[i]
            inc[j] += out[i] * s[i]
        spend = np.zeros(n)
        spend[idx] = (out[idx] * a[idx] * self.radio_bit[idx, p[idx]]
                      + E.erx(rx_bits[idx]) + E.erx(self.LC))
        broke = start_alive & (spend > self.E)
        blocked = np.zeros(n, bool)
        for i in idx:
            u, f = i, 1.0
            while u != n:
                if broke[u]:
                    blocked[i] = True
                    break
                f *= s[u]; u = p[u]
            frac[i] = f
        self.E[idx] -= spend[idx]
        self.E[broke] = 0.0

        b_succ = np.zeros(n); b_retry = np.zeros(n); b_fail = np.zeros(n)
        a_dir, s_dir = _arq(self.prr[np.arange(n), n])
        for i in idx:
            kb = own[i] / 1000.0
            if not blocked[i]:
                b_succ[i] = kb * frac[i]; b_fail[i] = kb * (1 - frac[i])
            else:
                cost = float(own[i] * a_dir[i] * self.radio_bit[i, n])
                if not broke[i] and self.E[i] >= cost:
                    self.E[i] -= cost
                    b_succ[i] = kb * s_dir[i]; b_retry[i] = kb * s_dir[i]; b_fail[i] = kb * (1 - s_dir[i])
                else:
                    b_fail[i] = kb
        self.E = self.E.clip(min=0.0)
        died = start_alive & (self.E <= 0)
        self.alive &= ~died
        self.round += 1
        consumed = float((start_energy[start_alive] - self.E[start_alive]).sum())
        return {"round": self.round, "b_succ": b_succ, "b_retry": b_retry, "b_fail": b_fail,
                "energy_consumed": consumed, "e_stand": 0.0, "n_alive_start": int(start_alive.sum()),
                "n_alive_end": int(self.alive.sum()), "n_dead": int(n - self.alive.sum()),
                "dead_frac": 1.0 - self.alive.mean(), "parent": p}
