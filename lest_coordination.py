"""
LEST — Lightweight State Table coordination mechanism
======================================================
Extracted from the WMN two-tier project for reuse.

The mechanism is independent of what quantity it tracks. Original used
per-node ENERGY; this file parameterises it so you can track LOAD (or
anything else monotonic per node) instead.

Four parts, all reusable:
  1. StateEntry / StateTable  — the table itself
  2. Tier assignment + hysteresis — when to signal
  3. Piggyback + heartbeat    — how state moves, O(n), no new packets
  4. Deterministic election   — zero-message leader selection from snapshot

HONEST NOTE ON PROVENANCE
-------------------------
This coordination mechanism was built for a project that closed. The
project closed because transmit-power control had a constant optimum,
not because the coordination mechanism was wrong — the mechanism was
never the failure point. But it was also never shown to *help*: in
closed-loop tests coordination moved the agent to parity with random,
not past it. So reuse the PATTERN, don't inherit the claim that it
improves anything. That has to be re-measured in your setting.

Also: the original `build_lest` counted ~639 tier transitions per node
while the rollout reported zero recalibrations. Those were two different
energy-normalisation schemes. `build_state_table_consistent()` below is
the reconciled version (20 transitions / 20,000 entries = 0.100%). Use
that one.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Callable


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# Signal size: 1 byte tier ID + 1 byte node ID + 4 bytes checksum
RECAL_SIGNAL_BYTES = 6

# Steps of missed heartbeat before nodes declare coordinator failed
HEARTBEAT_TIMEOUT = 5

# Hysteresis deadband — the normalised metric must move by at least
# this much ON TOP of crossing a boundary before a signal is issued.
# This is what stops a node sitting on a boundary from emitting a
# signal every single step. Tune it: too small and you get chatter,
# too large and tier changes are reported late.
HYSTERESIS_BAND = 0.05

# Tier boundaries on the normalised metric (0 = depleted/idle,
# 1 = full/saturated). Rename the tiers to suit your metric.
TIER_BOUNDS = [
    (0.75, 'High'),
    (0.40, 'Medium'),
    (0.15, 'Low'),
    (0.00, 'Critical'),
]


def assign_tier(norm_value: float) -> str:
    """Map a normalised [0,1] metric to a tier label."""
    for threshold, label in TIER_BOUNDS:
        if norm_value >= threshold:
            return label
    return TIER_BOUNDS[-1][1]


# ─────────────────────────────────────────────────────────────
# 1 — THE TABLE
# ─────────────────────────────────────────────────────────────

@dataclass
class StateEntry:
    """One row. `raw` is the cumulative/instantaneous quantity you
    track (energy consumed, packets relayed, queue occupancy...).
    `norm` is it mapped to [0,1]. `tier` is the discretised label
    that actually gets signalled — this is the compression that
    makes the mechanism O(n) and 1 byte per node."""
    node_id:      int
    raw:          float = 0.0
    norm:         float = 1.0
    tier:         str   = 'High'
    last_updated: int   = 0


class Coordinator:
    """
    Central coordinator maintaining the state table.

    The whole point: nodes never talk to each other. Each node reports
    its own scalar to the coordinator by piggybacking on a packet it was
    sending anyway (an ACK). The coordinator holds the global picture and
    broadcasts a snapshot on the heartbeat it was sending anyway. Net new
    packets: zero. Net new bytes: 2 per node per report + n per heartbeat.

    For a LOAD table, set `higher_is_worse=True` so that a heavily loaded
    node normalises toward 0 (same direction as a depleted battery) and
    the tier labels keep their meaning.
    """

    def __init__(self, node_ids, budget=1.0, higher_is_worse=True):
        self.node_ids = list(node_ids)
        self.table = {nid: StateEntry(node_id=nid) for nid in node_ids}
        self.budget = budget          # normalisation denominator
        self.higher_is_worse = higher_is_worse
        self.is_alive = True
        self.current_step = 0

        # Overhead accounting — report these, they are the cheap part
        # of the claim and the part reviewers check.
        self.signals_sent = 0
        self.signal_bytes = 0
        self.heartbeats_sent = 0
        self.heartbeat_bytes = 0
        self.report_bytes = 0
        self.signal_log = []

    # ── state update ────────────────────────────────────────
    def report(self, node_id, delta, cumulative=True):
        """
        Called when a node's piggybacked reading arrives.

        delta       : the increment (or absolute value if cumulative=False)
        cumulative  : True for energy-style accumulation, False for an
                      instantaneous reading like current queue depth.

        Returns a recalibration signal dict if the node crossed a tier
        boundary by more than the hysteresis band, else None.
        """
        e = self.table[node_id]
        if cumulative:
            e.raw += delta
        else:
            e.raw = delta
        e.last_updated = self.current_step
        self.report_bytes += 2        # 2-byte piggyback per report

        frac = e.raw / self.budget if self.budget else 0.0
        new_norm = 1.0 - frac if self.higher_is_worse else frac
        new_norm = float(np.clip(new_norm, 0.0, 1.0))

        new_tier = assign_tier(new_norm)
        old_norm, old_tier = e.norm, e.tier

        # Hysteresis: boundary crossing is necessary but not sufficient.
        crossed = (new_tier != old_tier and
                   abs(new_norm - old_norm) > HYSTERESIS_BAND)

        e.norm = new_norm
        if not crossed:
            return None

        e.tier = new_tier
        sig = {'node_id': node_id, 'new_tier': new_tier,
               'old_tier': old_tier, 'step': self.current_step,
               'bytes': RECAL_SIGNAL_BYTES}
        self.signals_sent += 1
        self.signal_bytes += RECAL_SIGNAL_BYTES
        self.signal_log.append(sig)
        return sig

    # ── broadcast ───────────────────────────────────────────
    def heartbeat(self):
        """One per step. Carries the snapshot (see snapshot())."""
        self.heartbeats_sent += 1
        self.heartbeat_bytes += len(self.node_ids)   # 1 byte tier/node
        self.current_step += 1

    def snapshot(self):
        """Full table copy, broadcast to every node on the heartbeat.

        THIS IS THE PIECE THAT MAKES ZERO-MESSAGE ELECTION WORK, and it
        is also the piece that is easy to forget to cost. It is
        per-step O(n) broadcast traffic. Count it (heartbeat_bytes
        above) — a reviewer will ask."""
        return {nid: StateEntry(e.node_id, e.raw, e.norm,
                                e.tier, e.last_updated)
                for nid, e in self.table.items()}

    def fail(self):
        self.is_alive = False

    def recover(self, snapshot=None):
        self.is_alive = True
        if snapshot:
            self.table.update(snapshot)

    # ── accounting ──────────────────────────────────────────
    def overhead_summary(self):
        slots = max(self.current_step * len(self.node_ids), 1)
        rate = self.signals_sent / slots * 100
        total = self.signal_bytes + self.heartbeat_bytes + self.report_bytes
        print("=== Coordination overhead ===")
        print(f"Steps                  : {self.current_step}")
        print(f"Nodes                  : {len(self.node_ids)}")
        print(f"Tier-change signals    : {self.signals_sent}")
        print(f"  signal bytes         : {self.signal_bytes}")
        print(f"  rate                 : {rate:.3f}% of (step x node) slots")
        print(f"Heartbeat snapshot bytes: {self.heartbeat_bytes}")
        print(f"Piggyback report bytes : {self.report_bytes}")
        print(f"TOTAL bytes            : {total}")
        print(f"  per step             : {total / max(self.current_step,1):.1f}")
        return {'signals': self.signals_sent, 'rate_pct': rate,
                'total_bytes': total}


# ─────────────────────────────────────────────────────────────
# 2 — NODE SIDE
# ─────────────────────────────────────────────────────────────

class CoordinatedNode:
    """
    Node holding a local copy of the table. It does NOT compute the
    table — it only receives snapshots. The only thing it computes
    locally is the election (see elect_leader), and that is why the
    election costs zero messages: every node runs the same argmax over
    the same snapshot and gets the same answer.
    """

    def __init__(self, node_id, on_tier_change: Optional[Callable] = None):
        self.node_id = node_id
        self.snapshot: Dict[int, StateEntry] = {}
        self.last_heartbeat = 0
        self.current_step = 0
        self.is_leader = False
        self.signals_received = 0
        # callback invoked with the new tier label — wire this to
        # whatever your node reconfigures (hyperparameters, duty cycle,
        # relay willingness, admission threshold...)
        self.on_tier_change = on_tier_change

    def receive_heartbeat(self, step, snapshot):
        self.last_heartbeat = step
        self.snapshot = snapshot

    def receive_signal(self, sig):
        if sig and sig['node_id'] == self.node_id:
            self.signals_received += 1
            if self.on_tier_change:
                self.on_tier_change(sig['new_tier'])

    def coordinator_alive(self):
        return (self.current_step - self.last_heartbeat) <= HEARTBEAT_TIMEOUT

    def my_tier(self):
        e = self.snapshot.get(self.node_id)
        return e.tier if e else 'High'

    def peer_tier(self, other_id):
        """This is the bit you said you want: every node knows every
        other node's state, without ever having asked any of them."""
        e = self.snapshot.get(other_id)
        return e.tier if e else None


# ─────────────────────────────────────────────────────────────
# 3 — DETERMINISTIC ELECTION (zero messages)
# ─────────────────────────────────────────────────────────────

def elect_leader(nodes, prefer='max'):
    """
    Every node runs this independently on its own snapshot copy and
    reaches the same answer, so no election traffic is needed.

    prefer='max' : pick the highest norm (most energy / least loaded)
    prefer='min' : pick the lowest  norm (most loaded) — e.g. if the
                   leader's job is to be relieved rather than to serve.

    PRECONDITION that is easy to violate: every node must hold the SAME
    snapshot. If snapshots can diverge (a node missed a heartbeat), the
    nodes can elect different leaders and you get a split brain. The
    original code assumed no divergence. If you need it to be safe,
    tag each snapshot with the step it came from and refuse to elect
    on a stale one — the check below does that.
    """
    if not nodes:
        return None

    # staleness guard — all electors must agree on snapshot vintage
    steps = {max((e.last_updated for e in n.snapshot.values()), default=-1)
             for n in nodes if n.snapshot}
    if len(steps) > 1:
        # snapshots diverged; a real deployment should fall back to a
        # messaged election here rather than silently split.
        return None

    any_snap = next((n.snapshot for n in nodes if n.snapshot), None)
    if not any_snap:
        return None

    pick = max if prefer == 'max' else min
    leader_id = pick(any_snap.keys(), key=lambda k: any_snap[k].norm)

    for n in nodes:
        n.is_leader = (n.node_id == leader_id)
    return leader_id


# ─────────────────────────────────────────────────────────────
# 4 — OFFLINE TABLE BUILD (the reconciled version)
# ─────────────────────────────────────────────────────────────

def build_state_table_consistent(df, node_col='NodeId',
                                 episode_col='Episode',
                                 step_col='Step',
                                 metric_col='EnergyConsumed',
                                 n_episodes=None,
                                 higher_is_worse=True):
    """
    Build the table offline from a trace, with ONE normalisation scheme
    used everywhere.

    This is the version that resolved the contradiction in the original
    project: the earlier build normalised per (episode, node) against a
    global max, which re-tiered every node every episode and produced
    ~639 "transitions" per node, while the online rollout used a
    different scheme and reported zero. Same data, two incompatible
    numbers, and the figure and the text disagreed.

    Here: one cumulative series per node across the whole trace, divided
    by one budget. Gave 20 transitions over 20,000 (episode x node)
    entries = 0.100%.

    If you reuse this, make sure the offline build and the online
    Coordinator.report() use the SAME budget and the SAME direction,
    or you will reproduce the same contradiction in your own paper.
    """
    d = df.sort_values([node_col, episode_col, step_col]).copy()
    d['_cum'] = d.groupby(node_col)[metric_col].cumsum()

    per_ep_max = d.groupby([episode_col, node_col])[metric_col].sum().max()
    n_eps = n_episodes or d[episode_col].nunique()
    budget = per_ep_max * n_eps

    frac = d['_cum'] / budget
    d['_norm'] = (1.0 - frac if higher_is_worse else frac).clip(0, 1)
    d['_tier'] = d['_norm'].apply(assign_tier)

    table = (d.groupby([episode_col, node_col])
               .agg(raw=('_cum', 'last'),
                    norm=('_norm', 'last'),
                    tier=('_tier', 'last'))
               .reset_index())

    t = table.sort_values([node_col, episode_col])
    t['_prev'] = t.groupby(node_col)['tier'].shift(1)
    transitions = int((t['tier'] != t['_prev']).sum() - t[node_col].nunique())
    transitions = max(transitions, 0)

    print(f"Entries     : {len(table):,}")
    print(f"Transitions : {transitions}")
    print(f"Rate        : {transitions / max(len(table),1) * 100:.3f}%")
    print(table['tier'].value_counts().to_string())

    return table, {'entries': len(table), 'transitions': transitions,
                   'rate_pct': transitions / max(len(table), 1) * 100,
                   'budget': budget}


# ─────────────────────────────────────────────────────────────
# ADAPTING THIS TO A LOAD TABLE
# ─────────────────────────────────────────────────────────────
"""
To track LOAD instead of energy, three changes:

1. metric_col -> whatever measures load. Relay count and queue depth
   behave differently: relay count accumulates (cumulative=True),
   queue depth is instantaneous (cumulative=False). Pick one and be
   consistent; mixing them is how the original contradiction happened.

2. budget -> what counts as "fully loaded". For queue depth this is
   buffer capacity and is a real hardware number. For relay count
   there is no natural ceiling, so you are picking a scale factor —
   say so in the paper rather than letting it look derived.

3. Tier labels. 'High/Medium/Low/Critical' read as "lots of headroom"
   -> "none" with higher_is_worse=True. If that inverts in your
   setting, rename them rather than flipping the direction silently.

One thing worth measuring before you build on it: whether the tiers
carry information. In the original project the agent's state space
collapsed — 65-81% of all experience landed in 3 of 36 cells, so most
of the table was never exercised. Check the tier occupancy histogram
early. If every node sits in one tier for the whole run (which is what
happened across all 9 hardware configurations — all Medium), the table
is correct but is not discriminating anything, and anything downstream
that consumes the tier is receiving a constant.
"""
