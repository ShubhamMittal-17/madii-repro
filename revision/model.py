"""Deployments and control accounting for the revision experiments.

Accounting (identical for every router, so it adds no bias between them):
  - data packet   L  = 4000 payload + 16-bit battery report piggybacked by the source
                       (every energy-aware router, the LP planner and MADII's state upload
                       need the residual battery at the sink; it is now paid for, not assumed free)
  - advertisement LC = 100-bit header + n * ceil(log2(n + 1)) bits, so that every node's
                       next-hop identifier is actually encoded (800 bits at n = 100)
The legacy accounting (L = 4000, LC = 100) is available for comparison with earlier results.

Layouts: uniform (as MADII) or clustered (Gaussian clusters); the sink at the field centre
(as MADII) or at a corner.
"""
import math

import numpy as np

import env as E

PAYLOAD = 4000
BATTERY_REPORT_BITS = 16
ADV_HEADER_BITS = 100


def advert_bits(n):
    return ADV_HEADER_BITS + n * math.ceil(math.log2(n + 1))


def positions(n, side, seed, layout="uniform", n_clusters=5, spread=0.08):
    rng = np.random.default_rng(seed)
    if layout == "uniform":
        return rng.uniform(0, side, size=(n, 2))
    if layout == "clustered":
        centres = rng.uniform(0.1, 0.9, size=(n_clusters, 2)) * side
        k = rng.integers(n_clusters, size=n)
        p = centres[k] + rng.normal(0, spread * side, size=(n, 2))
        return np.clip(p, 0, side)
    raise ValueError(layout)


class Deployment(E.WSN):
    """MADII's WSN with a configurable layout, sink position and control accounting.

    With layout="uniform", sink="center" and accounting="legacy" it is identical to env.WSN
    (same positions for the same seed), which test_revision.py checks."""

    def __init__(self, n=100, side=500.0, e0=0.5, seed=0, traffic=None, layout="uniform",
                 sink="center", accounting="revised"):
        if accounting == "revised":
            L, LC = PAYLOAD + BATTERY_REPORT_BITS, advert_bits(n)
        elif accounting == "legacy":
            L, LC = PAYLOAD, ADV_HEADER_BITS
        else:
            raise ValueError(accounting)
        super().__init__(n=n, side=side, e0=e0, packet_bits=L, control_bits=LC, seed=seed, traffic=traffic)
        self.payload_bits = PAYLOAD
        self.accounting, self.layout, self.sink_at = accounting, layout, sink
        if layout != "uniform" or sink != "center":
            pos = positions(n, side, seed, layout)
            s = {"center": [side / 2, side / 2], "corner": [0.0, 0.0]}[sink]
            self.set_positions(np.vstack([pos, [s]]))

    def set_positions(self, pos):
        self.pos = np.asarray(pos, float)
        self.d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
        self.d2s = self.d[:self.n, self.n]
        self.etx_bit = E.etx(1.0, self.d)
