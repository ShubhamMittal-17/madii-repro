"""MADII's simulation world (Yang et al., IEEE IoT-J 12(23), 2025), Eq. (1)-(2), (8)-(22).

Faithful to Table I: 100 SNs uniform in 500x500 m, sink at (250,250), 4 kbit data
packet, 0.1 kbit control packet, E0 = 0.5 J, Eelec 50 nJ/bit, eps_fs 10 pJ/bit/m^2,
eps_amp 0.0013 pJ/bit/m^4. No data fusion when relaying (paper Sec. IV-A-2 item 4).

Round mechanics the paper leaves implicit, and our choice for each:
  R1 A relay that cannot afford its send DIES INSIDE the round; everything routed
     through it fails that round. Evidence: Table VI's DDV is short of SR*400 kbit
     by exactly one packet at FND for most methods, so the dying node's packet is
     lost. (The alternative -- deaths only at round boundaries -- loses nothing.)
  R2 Failed sources then retransmit DIRECTLY to the sink if they can afford it
     (the paper's "Routing Reissuing", STCIN3C step 6); otherwise the packet fails.
  R3 A next hop that is dead, or a cycle, is resolved the same way: send direct.
  R4 Every alive node pays to receive the sink's RouteADV (0.1 kbit) each round.
     Their per-round state upload rides along with the data packet, so it is not
     charged separately.
"""
import numpy as np

E_ELEC, EPS_FS, EPS_AMP = 50e-9, 10e-12, 0.0013e-12
D0 = np.sqrt(EPS_FS / EPS_AMP)          # 87.71 m


def etx(bits, dist):
    bits = np.asarray(bits, float); dist = np.asarray(dist, float)
    amp = np.where(dist < D0, EPS_FS * dist ** 2, EPS_AMP * dist ** 4)
    return bits * E_ELEC + bits * amp


def erx(bits):
    return np.asarray(bits, float) * E_ELEC


class WSN:
    """One `step(parent)` is one data-collection round. parent[i] in [0..n]; n = sink."""

    def __init__(self, n=100, side=500.0, e0=0.5, packet_bits=4000,
                 control_bits=100, seed=0, aggregate=False, traffic=None):
        self.n, self.side, self.e0 = n, side, e0
        # traffic[t, i] = packets node i generates in round t (dynamic-traffic bench);
        # None = one packet per node per round, the paper's setting.
        self.traffic = None if traffic is None else np.asarray(traffic, int)
        self.aggregate = aggregate      # True = relays fuse to one packet (calibration only)
        self.L, self.LC = packet_bits, control_bits
        rng = np.random.default_rng(seed)
        self.pos = np.vstack([rng.uniform(0, side, size=(n, 2)),
                              [[side / 2, side / 2]]])          # index n = sink
        self.d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
        self.d2s = self.d[:n, n]
        self.etx_bit = etx(1.0, self.d)                          # per-bit TX cost matrix
        self.reset()

    def reset(self):
        self.E = np.full(self.n, self.e0)
        self.alive = np.ones(self.n, bool)
        self.round = 0
        return self.state()

    def state(self):
        return {"pos": self.pos[:self.n], "energy": self.E.copy(),
                "alive": self.alive.copy(), "d2s": self.d2s, "round": self.round}

    # -- routing helpers ----------------------------------------------------
    def _sanitize(self, parent):
        """R3: point invalid/dead/self/cyclic next hops at the sink."""
        n = self.n
        p = np.asarray(parent, int).copy()
        bad = (p < 0) | (p > n) | (np.arange(n) == p)
        p[bad] = n
        dead_hop = (p < n) & ~self.alive[np.clip(p, 0, n - 1)]
        p[dead_hop] = n
        # break cycles: walk each alive node's path, cap at n steps
        for i in np.flatnonzero(self.alive):
            seen, u = set(), i
            while u != n:
                if u in seen or not self.alive[u]:
                    p[i] = n
                    break
                seen.add(u)
                u = p[u]
        p[~self.alive] = n
        return p

    def _loads(self, p):
        """bits each node transmits, given the forest p (alive nodes only)."""
        n = self.n
        load = self.own_bits()
        if self.aggregate:
            return load                                         # fusion: forward one packet
        order = np.argsort(-self._depth(p))                     # deepest first
        for i in order:
            if self.alive[i] and p[i] != n:
                load[p[i]] += load[i]
        return load

    def _depth(self, p):
        n, depth = self.n, np.zeros(self.n, int)
        for i in range(self.n):
            if not self.alive[i]:
                continue
            u, k = i, 0
            while u != n and k < n:
                u = p[u]; k += 1
            depth[i] = k
        return depth

    def own_bits(self):
        """Bits each alive node generates this round."""
        if self.traffic is None:
            return np.where(self.alive, float(self.L), 0.0)
        k = self.traffic[min(self.round, len(self.traffic) - 1)]
        return np.where(self.alive, k * float(self.L), 0.0)

    # -- one round ----------------------------------------------------------
    def step(self, parent):
        n, L = self.n, self.L
        own = self.own_bits()
        p = self._sanitize(parent)
        start_alive = self.alive.copy()
        n_start = int(start_alive.sum())

        load = self._loads(p)
        if self.aggregate:                                       # received bits (pre-fusion)
            rx = np.zeros(n)
            for i in np.flatnonzero(start_alive):
                if p[i] != n:
                    rx[p[i]] += own[i]
        else:
            rx = np.where(start_alive, load - own, 0.0).clip(min=0)
        spend = np.zeros(n)
        idx = np.flatnonzero(start_alive)
        spend[idx] = (load[idx] * self.etx_bit[idx, p[idx]] + erx(rx[idx])
                      + erx(self.LC))                            # R4

        broke = start_alive & (spend > self.E)                   # R1: dies mid-round
        # a source succeeds iff no node on its path (itself included) broke
        blocked = np.zeros(n, bool)
        for i in idx:
            u = i
            while u != n:
                if broke[u]:
                    blocked[i] = True
                    break
                u = p[u]
        # charge the attempt; broke nodes are drained
        self.E[idx] -= spend[idx]
        self.E[broke] = 0.0

        b_succ = np.zeros(n); b_retry = np.zeros(n); b_fail = np.zeros(n)
        for i in idx:
            kb = own[i] / 1000.0
            if not blocked[i]:
                b_succ[i] = kb
            else:                                                # R2: retry direct
                cost = float(etx(own[i], self.d2s[i]))
                if not broke[i] and self.E[i] >= cost:
                    self.E[i] -= cost
                    b_succ[i] = kb; b_retry[i] = kb
                else:
                    b_fail[i] = kb

        self.E = self.E.clip(min=0.0)
        died = start_alive & (self.E <= 0)
        self.alive &= ~died
        self.round += 1

        consumed = float(self.e0 * n_start - self.E[start_alive].sum())
        e_stand = float(etx(own[start_alive], self.d2s[start_alive]).sum())
        return {"round": self.round, "b_succ": b_succ, "b_retry": b_retry,
                "b_fail": b_fail, "energy_consumed": consumed, "e_stand": e_stand,
                "n_alive_start": n_start, "n_alive_end": int(self.alive.sum()),
                "n_dead": int(self.n - self.alive.sum()),
                "dead_frac": 1.0 - self.alive.mean(), "parent": p}


W_SUCC = W_FAIL = 10.0
W_RETRY = 20.0


def round_reward(info):
    """Eq. (8): (R_succ - P_fail - P_retry) * (E_stand / E_consumed) * first-try ratio."""
    n_succ = info["b_succ"].sum(); n_retry = info["b_retry"].sum()
    r_valid = W_SUCC * n_succ - W_FAIL * info["b_fail"].sum() - W_RETRY * n_retry
    e_rel = info["e_stand"] / info["energy_consumed"] if info["energy_consumed"] > 0 else 0.0
    ratio = (n_succ - n_retry) / n_succ if n_succ > 0 else 0.0
    return float(r_valid * e_rel * ratio)


def run_episode(env, policy, betas=(0.01, 0.10, 0.25, 0.50), max_rounds=400):
    """Run to the last beta. Returns {beta: {SR, RS, EE, DDV}} using Eq. (19)-(22)."""
    env.reset()
    n0 = env.n * env.e0
    ddv = rs = 0.0
    out, target = {}, max(betas)
    for _ in range(max_rounds):
        parent = policy(env)
        info = env.step(parent)
        ddv += float(info["b_succ"].sum())
        rs += round_reward(info)
        used = n0 - float(env.E.sum())
        for b in betas:
            if b not in out and info["n_dead"] >= np.ceil(b * env.n):   # integer compare
                out[b] = {"SR": info["round"], "RS": rs, "DDV": ddv,
                          "EE": ddv / used if used > 0 else 0.0}
        if info["dead_frac"] >= target or info["n_alive_end"] == 0:
            break
    for b in betas:
        out.setdefault(b, {"SR": env.round, "RS": rs, "DDV": ddv,
                           "EE": ddv / max(n0 - float(env.E.sum()), 1e-12)})
    return out
