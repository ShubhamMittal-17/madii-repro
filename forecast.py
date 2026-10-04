"""Per-node traffic forecasters for predictive routing (Part 3).

All share one interface, vectorised over nodes:
    f = Forecaster(n);  f.fit_history(hist)      # hist: (rounds, n) past traffic
    f.update(y)                                   # y: (n,) traffic observed this round
    mean, upper = f.forecast(H)                   # each (H, n), packets per round

The upper band is mean + z * (running mean absolute one-step error) * sqrt(h), so the
battery reserve can cover plausible bursts, not just the expected load.
No forecaster ever sees a round before it has happened.
"""
import numpy as np

from traffic import P


class _Base:
    def __init__(self, n, z=1.0, err_decay=0.1):
        self.n, self.z, self.err_decay = n, z, err_decay
        self.mae = np.ones(n)

    def _track_error(self, y):
        pred = self.forecast(1)[0][0]
        self.mae = (1 - self.err_decay) * self.mae + self.err_decay * np.abs(y - pred)

    def fit_history(self, hist):
        for y in hist:
            self.update(y)

    def _band(self, mean):
        h = np.arange(1, len(mean) + 1)[:, None]
        mean = np.clip(mean, 0, None)
        return mean, mean + self.z * self.mae[None, :] * np.sqrt(h)


class Persistence(_Base):
    """Next rounds = this round."""
    def __init__(self, n, **kw):
        super().__init__(n, **kw); self.last = np.ones(n)

    def update(self, y):
        self._track_error(y); self.last = np.asarray(y, float)

    def forecast(self, H):
        return self._band(np.repeat(self.last[None, :], H, axis=0))


class SeasonalNaive(_Base):
    """Next rounds = the same hour yesterday."""
    def __init__(self, n, **kw):
        super().__init__(n, **kw); self.buf = np.ones((P, n)); self.t = 0

    def update(self, y):
        self._track_error(y); self.buf[self.t % P] = y; self.t += 1

    def forecast(self, H):
        idx = (self.t + np.arange(H)) % P
        return self._band(self.buf[idx])


class HoltWinters(_Base):
    """Additive Holt-Winters with season length P (one day):
        level_t  = a (y_t - s_{t-P}) + (1-a)(level + trend)
        trend_t  = b (level_t - level_{t-1}) + (1-b) trend
        s_t      = g (y_t - level_t) + (1-g) s_{t-P}
        yhat_t+h = level + h trend + s_{t+h-P}
    The first season of history initialises level and season; trend starts at 0."""

    def __init__(self, n, alpha=0.2, beta=0.01, gamma=0.3, **kw):
        super().__init__(n, **kw)
        self.a, self.b, self.g = alpha, beta, gamma
        self.level = np.ones(n); self.trend = np.zeros(n); self.season = np.zeros((P, n)); self.t = 0

    def fit_history(self, hist):
        hist = np.asarray(hist, float)
        if len(hist) >= P:
            self.level = hist[:P].mean(0)
            self.season = hist[:P] - self.level
            self.t = P
            hist = hist[P:]
        for y in hist:
            self.update(y)

    def update(self, y):
        y = np.asarray(y, float)
        self._track_error(y)
        s_old = self.season[self.t % P]
        lv = self.a * (y - s_old) + (1 - self.a) * (self.level + self.trend)
        self.trend = self.b * (lv - self.level) + (1 - self.b) * self.trend
        self.season[self.t % P] = self.g * (y - lv) + (1 - self.g) * s_old
        self.level = lv
        self.t += 1

    def forecast(self, H):
        h = np.arange(1, H + 1)[:, None]
        mean = self.level[None, :] + h * self.trend[None, :] + self.season[(self.t + np.arange(H)) % P]
        return self._band(mean)


class HistoryMean(_Base):
    """No prediction: every future round = the node's mean rate over the deployment history,
    never updated. The baseline that tells whether a forecaster adds anything to a planner."""
    def __init__(self, n, **kw):
        super().__init__(n, **kw); self.m = np.ones(n)

    def fit_history(self, hist):
        self.m = np.asarray(hist, float).mean(0)

    def update(self, y):
        self._track_error(y)

    def forecast(self, H):
        return self._band(np.repeat(self.m[None, :], H, axis=0))


class Perfect(_Base):
    """Knows the true future traffic. A diagnostic ceiling for what any forecaster could add."""
    def __init__(self, n, future=None, **kw):
        super().__init__(n, **kw); self.fut = np.asarray(future, float); self.t = 0

    def fit_history(self, hist):
        pass

    def update(self, y):
        self.t += 1

    def forecast(self, H):
        m = self.fut[self.t:self.t + H]
        if len(m) < H:
            m = np.vstack([m, np.repeat(m[-1:], H - len(m), 0)])
        return m, m


FORECASTERS = {"perfect": Perfect, "holt-winters": HoltWinters, "seasonal-naive": SeasonalNaive, "persistence": Persistence,
               "history-mean": HistoryMean}
