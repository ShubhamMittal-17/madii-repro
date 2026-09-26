"""DQN training for the MADII reproduction (Algorithm 1 of the paper).

Two stages, as in their Algorithm 1:
  Stage 1 (imitation): the replay buffer is filled with transitions from a clustering
      algorithm (they use FCM / HBA / POA; we use FCM with cluster count drawn from
      3-18, their Table IV), and the network is trained on them.
  Stage 2 (exploration/exploitation): epsilon-greedy from max_epsilon to min_epsilon.

DQN settings from their Table III: gamma 0.99, epsilon 1 -> 0.1, target network updated
every C1 = 10 steps. Table II: lr 1e-4, batch 64.

Choices the paper does not specify (documented, ours):
  C1 One transition per ROUND: the joint action of all sensors, one shared reward.
  C2 Credit assignment: every alive sensor's CHOSEN Q value is regressed on the same
     shared round return (independent Q-learners with a common reward). An earlier
     version averaged the per-sensor Q values BEFORE the loss; that let the network
     meet the target with any mix of per-sensor values and it collapsed to "every
     sensor transmits direct to the sink" (measured: sink share 1.00). The paper does
     not say how its per-sensor Q table (Fig. 3) becomes one training target.
  C2b Double DQN: the online network picks the next action, the target network scores
     it. Plain DQN over 101 candidates per sensor overestimates badly.
  C3 Invalid next hops (self, dead, cycles) are masked at action time; anything that
     still forms a cycle falls back to a direct send, which is their own reissuing rule.
  C4 Replay capacity 10000 transitions; episode = one deployment run to first node death.
  C5 Reward scaled by 1e-3 for optimisation only (does not change any reported metric).
  C6 Epsilon is per sensor and all 100 act at once, so the paper's epsilon = 1.0 would
     randomise the entire network every round (every episode then dies in one round --
     measured). We keep their schedule's SHAPE but scale it to 0.2 -> 0.02, decaying
     exponentially. Stated as a deviation.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import env as E
import policies as P
import experts as X
import dijkstra_rl as D
from qnet import QNet, N_FEAT

DEV = torch.device("cpu")
REW_SCALE = 1e-3
DIAG = math.hypot(500.0, 500.0)


def features(w):
    n = w.n
    f = np.zeros((n + 1, N_FEAT), np.float32)
    f[:, 0] = w.pos[:, 0] / w.side
    f[:, 1] = w.pos[:, 1] / w.side
    f[:n, 2] = w.E / w.e0
    f[n, 2] = 1.0
    f[:n, 3] = w.d2s / DIAG
    f[:n, 4] = w.alive
    f[n, 4] = 1.0
    f[n, 5] = 1.0
    return f


def action_mask(w):
    """(n, n+1) bool: True where node i may pick j."""
    n = w.n
    m = np.zeros((n, n + 1), bool)
    m[:, :n] = w.alive[None, :]
    m[:, n] = True
    np.fill_diagonal(m[:, :n], False)
    return m


@torch.no_grad()
def greedy_action(net, w, eps=0.0, rng=None):
    q = net(torch.from_numpy(features(w)).unsqueeze(0)).squeeze(0).numpy()
    m = action_mask(w)
    q = np.where(m, q, -np.inf)
    a = q.argmax(1)
    if eps > 0:
        rng = rng or np.random
        rnd = rng.random(w.n) < eps
        if rnd.any():
            for i in np.flatnonzero(rnd & w.alive):
                cand = np.flatnonzero(m[i])
                a[i] = cand[rng.integers(len(cand))]
    a[~w.alive] = w.n
    return a.astype(np.int64)


class Buffer:
    def __init__(self, cap, n):
        self.cap, self.n, self.i, self.full = cap, n, 0, False
        self.s = np.zeros((cap, n + 1, N_FEAT), np.float32)
        self.s2 = np.zeros((cap, n + 1, N_FEAT), np.float32)
        self.a = np.zeros((cap, n), np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.al = np.zeros((cap, n), bool)
        self.m2 = np.zeros((cap, n, n + 1), bool)

    def add(self, s, a, r, s2, done, alive, mask2):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i] = s, a, r, s2
        self.d[i], self.al[i], self.m2[i] = done, alive, mask2
        self.i = (i + 1) % self.cap
        self.full |= self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, bs, rng):
        idx = rng.integers(0, len(self), bs)
        t = lambda x: torch.from_numpy(x[idx])
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d), t(self.al), t(self.m2)


def td_loss(net, s, a, alive, y, lossf):
    """C2: each alive sensor's chosen Q is regressed on the shared round return."""
    q = net(s)                                     # (B, n, n+1)
    chosen = q.gather(2, a.unsqueeze(-1)).squeeze(-1)
    w = alive.float()
    per = lossf(chosen, y.unsqueeze(1).expand_as(chosen))
    return (per * w).sum() / w.sum().clamp(min=1)


@torch.no_grad()
def target_value(net, tgt, s2, mask2, alive2):
    """C2b: Double DQN -- online net argmax, target net value, averaged over sensors."""
    a2 = net(s2).masked_fill(~mask2, -1e9).argmax(-1)
    q = tgt(s2).gather(2, a2.unsqueeze(-1)).squeeze(-1)
    w = alive2.float()
    return (q * w).sum(1) / w.sum(1).clamp(min=1)


def evaluate(net, seeds, betas=(0.01, 0.10, 0.25, 0.50), max_rounds=400, n=100):
    net.eval()
    rows = []
    for s in seeds:
        w = E.WSN(n=n, seed=s)
        rows.append(E.run_episode(w, lambda ww: greedy_action(net, ww), betas, max_rounds))
    net.train()
    return {b: {k: float(np.mean([r[b][k] for r in rows])) for k in ("SR", "RS", "EE", "DDV")}
            for b in betas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="transformer", choices=["transformer", "informer"])
    ap.add_argument("--tag", default="madti")
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--il-episodes", type=int, default=150)
    ap.add_argument("--experts", default="fcm", choices=["fcm", "mixed", "battery"],
                    help="mixed = FCM/HBA/POA in rotation, as the paper's stage 1 does; "
                         "battery = battery-weighted Dijkstra (Dijkstra-warm-started DQN, ours)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=10000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target-every", type=int, default=10)
    ap.add_argument("--eps-max", type=float, default=0.2)
    ap.add_argument("--eps-min", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=0.01, help="episode ends at this dead fraction")
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--max-rounds", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    Path("checkpoints").mkdir(exist_ok=True); Path("logs").mkdir(exist_ok=True)
    log = open(f"logs/{args.tag}.jsonl", "a")

    net = QNet(args.d_model, args.heads, args.d_ff, args.layers, args.arch).to(DEV)
    tgt = QNet(args.d_model, args.heads, args.d_ff, args.layers, args.arch).to(DEV)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buf = Buffer(args.buffer, args.nodes)
    lossf = nn.SmoothL1Loss(reduction="none")

    steps, t0, recent, best_sr = 0, time.time(), [], -1.0
    for ep in range(args.episodes):
        seed = int(rng.integers(1000, 100000))          # training pool, disjoint from eval
        w = E.WSN(n=args.nodes, seed=seed)
        imitating = ep < args.il_episodes
        k_fcm = int(rng.integers(3, 19))                # their Table IV: 3-18
        frac = max(0.0, (ep - args.il_episodes) / max(1, args.episodes - args.il_episodes))
        eps = args.eps_max * (args.eps_min / args.eps_max) ** frac          # C6
        ep_r = 0.0
        for _ in range(args.max_rounds):
            s = features(w)
            alive = w.alive.copy()
            if imitating:
                if args.experts == "battery":
                    a = D.fast_battery_weighted(w).astype(np.int64)
                elif args.experts == "mixed":
                    which = ep % 3
                    a = (P.fcm_routing(w, k=k_fcm, rng=rng) if which == 0 else
                         X.hba_routing(w, rng=rng) if which == 1 else
                         X.poa_routing(w, rng=rng)).astype(np.int64)
                else:
                    a = P.fcm_routing(w, k=k_fcm, rng=rng).astype(np.int64)
            else:
                a = greedy_action(net, w, eps, rng)
            info = w.step(a)
            r = E.round_reward(info) * REW_SCALE
            ep_r += r
            done = float(info["n_dead"] >= math.ceil(args.beta * w.n))
            buf.add(s, a, r, features(w), done, alive, action_mask(w))
            if len(buf) >= max(args.batch, 500):
                bs, ba, br, bs2, bd, bal, bm2 = buf.sample(args.batch, rng)
                y = br + args.gamma * (1 - bd) * target_value(net, tgt, bs2, bm2,
                                                              bs2[:, :-1, 4] > 0.5)
                loss = td_loss(net, bs, ba, bal, y, lossf)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                steps += 1
                if steps % args.target_every == 0:
                    tgt.load_state_dict(net.state_dict())
            if done or info["n_alive_end"] == 0:
                break
        recent.append(w.round)
        if (ep + 1) % 10 == 0:
            rec = {"ep": ep + 1, "stage": "IL" if imitating else "eps",
                   "eps": round(eps, 3), "fnd_mean_last10": float(np.mean(recent[-10:])),
                   "ep_reward": round(ep_r, 2), "steps": steps,
                   "mins": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes:
            ev = evaluate(net, range(args.eval_seeds), n=args.nodes)
            rec = {"ep": ep + 1, "eval": {str(b): {k: round(v, 1) for k, v in d.items()}
                                          for b, d in ev.items()}}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"model": net.state_dict(), "args": vars(args), "ep": ep + 1},
                       f"checkpoints/{args.tag}.pt")
            if ev[0.01]["SR"] > best_sr:                    # keep the best, not the last
                best_sr = ev[0.01]["SR"]
                torch.save({"model": net.state_dict(), "args": vars(args), "ep": ep + 1,
                            "eval_sr": best_sr}, f"checkpoints/{args.tag}_best.pt")
    torch.save({"model": net.state_dict(), "args": vars(args), "ep": args.episodes},
               f"checkpoints/{args.tag}.pt")


if __name__ == "__main__":
    main()
