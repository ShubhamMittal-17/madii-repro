"""Synthetic time-varying traffic for the dynamic bench (Part 3).

Each generator returns (history, traffic): integer packets per node per round, shape
(rounds, n). `history` is the traffic of the days before deployment, which the forecaster
may learn from (a real network has a past); `traffic` is what happens after deployment.
One "day" is P = 24 rounds.

  R  "forecast right": a spatial hotspot of nodes surges every evening, Poisson noise on top.
  N  "not needed":     every node sends exactly one packet per round (the paper's setting).
  B  "sudden burst":   the R pattern plus unforecastable bursts: with probability p_burst per
                       round a random node (any node, relays included) sends M x its rate for
                       D rounds.
  D  "pattern shift":  the R pattern, but `shift_day` days into the deployment the hotspot
                       moves to a new random spot (a different 20% of nodes) for good. The
                       history only ever shows the old hotspot, so a fixed historical rate is
                       wrong from then on; a forecaster has to notice and adapt.

Every generator is deterministic given its seed.
"""
import numpy as np

P = 24                 # rounds per day
HIST_DAYS = 3          # days of pre-deployment history given to the forecaster


def _hotspot(pos, rng, frac=0.2):
    """The `frac` of nodes nearest a random field point form the hotspot."""
    c = rng.uniform(0.15, 0.85, 2) * pos.max()
    d = np.linalg.norm(pos - c, axis=1)
    return np.argsort(d)[: int(round(frac * len(pos)))]


def _pattern_rates(n, T, hot, peak=4.0, off=0.5, window=6, phase=17):
    """Expected packets per round: 1 for ordinary nodes; for hotspot nodes `peak` during a
    `window`-round evening starting at hour `phase`, else `off`."""
    rate = np.ones((T, n))
    hour = np.arange(T) % P
    evening = (hour >= phase) & (hour < phase + window)
    rate[:, hot] = np.where(evening[:, None], peak, off)
    return rate


def generate(scenario, pos, seed, T=400, p_burst=0.05, burst_x=5.0, burst_len=4, shift_day=2):
    rng = np.random.default_rng(seed)
    n = len(pos)
    H = HIST_DAYS * P
    if scenario == "N":
        ones = np.ones((H + T, n), int)
        return ones[:H], ones[H:]
    hot = _hotspot(pos, rng)
    rate = _pattern_rates(n, H + T, hot)
    if scenario == "D":
        t0 = H + shift_day * P
        rate[t0:] = _pattern_rates(n, H + T, _hotspot(pos, np.random.default_rng(seed + 7919)))[t0:]
    if scenario == "B":
        t = 0
        while t < H + T:
            if rng.random() < p_burst:
                i = rng.integers(n)
                rate[t:t + burst_len, i] *= burst_x
            t += 1
    elif scenario not in ("R", "D"):
        raise ValueError(scenario)
    k = rng.poisson(rate)
    return k[:H], k[H:]
