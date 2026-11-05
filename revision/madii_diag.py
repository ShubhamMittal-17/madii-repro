"""Why does the corrected MADII reconstruction collapse to direct transmission? (review M4)

Scores fixed policies with MADII's corrected per-round reward (Eq. 8, env.round_reward) on the
30 static test deployments: is direct-to-sink what the reward prefers, or a learning failure?
No neural network is needed. Writes results/revision/madii_diag.json.
"""
import json

import numpy as np

import dijkstra_rl as D
import env as E
import policies as P

POLICIES = {"Direct to sink": P.direct, "Min-energy Dijkstra": P.min_energy,
            "Battery-weighted Dijkstra": D.fast_battery_weighted}


def episode(seed, pol, gamma=0.99, max_rounds=400):
    w = E.WSN(seed=seed)
    rew, F = [], None
    while w.round < max_rounds:
        info = w.step(pol(w))
        rew.append(E.round_reward(info))
        if F is None and info["n_dead"] >= 1:
            F = info["round"]
        if info["dead_frac"] >= 0.5:
            break
    rew = np.array(rew)
    return dict(F=F, first_round=float(rew[0]), mean_to_F=float(rew[:F].mean()),
                discounted=float((rew * gamma ** np.arange(len(rew))).sum()), rounds=len(rew))


def main():
    out = {}
    for name, pol in POLICIES.items():
        rows = [episode(s, pol) for s in range(30)]
        out[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[name]["per_seed_first_round"] = [r["first_round"] for r in rows]
        print(f"{name:28s} FND {out[name]['F']:6.1f}  reward round 1 {out[name]['first_round']:9.1f}  "
              f"mean to FND {out[name]['mean_to_F']:9.1f}  discounted {out[name]['discounted']:11.1f}")
    json.dump(out, open("results/revision/madii_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
