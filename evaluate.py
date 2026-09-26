"""Head-to-head table: our reproduced learner vs classical routing vs the LP ceiling,
with MADII's own printed numbers alongside. Run after training.

  .venv/bin/python evaluate.py --ckpt checkpoints/madti.pt --seeds 30
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

import env as E, policies as P
from lp_bound import max_lifetime_T
from qnet import QNet
from train import greedy_action

PAPER = {  # Table VI, (SR, EE, DDV) at beta = 1/10/25/50%
    "MADII (printed)":  [(19, 817, 7596), (37, 833, 14216), (52, 723, 19244), (89, 742, 25708)],
    "MADTI (printed)":  [(12, 352, 4792), (17, 372, 6664), (28, 418, 10308), (67, 580, 19692)],
    "FCM (printed)":    [(19, 555, 7596), (27, 533, 10636), (32, 531, 12228), (57, 597, 18704)],
}
BETAS = (0.01, 0.10, 0.25, 0.50)


def load(ckpt):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = c["args"]
    net = QNet(a["d_model"], a["heads"], a["d_ff"], a["layers"], a["arch"])
    net.load_state_dict(c["model"]); net.eval()
    return net, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=["checkpoints/madti.pt"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=100)
    args = ap.parse_args()
    seeds = range(args.seeds)

    ceil = {s: max_lifetime_T(E.WSN(n=args.nodes, seed=s)) for s in seeds}
    arms = {"direct": P.direct, "FCM k=10": P.fcm_routing,
            "min-energy Dijkstra": P.min_energy,
            "battery Dijkstra a=1": P.battery_weighted}
    for c in args.ckpt:
        net, a = load(c)
        name = f"{Path(c).stem} ({a['arch']}, {'IL' if a['il_episodes'] else 'no IL'})"
        arms[name] = (lambda nn_: (lambda w: greedy_action(nn_, w)))(net)

    res = {}
    for name, pol in arms.items():
        runs = [E.run_episode(E.WSN(n=args.nodes, seed=s), pol, BETAS) for s in seeds]
        res[name] = {b: {k: np.array([r[b][k] for r in runs]) for k in ("SR", "EE", "DDV", "RS")}
                     for b in BETAS}
        # rounds fully completed before the first death, against the LP ceiling
        res[name]["pct"] = np.array([(r[0.01]["SR"] - 1) / ceil[s] * 100 for r, s in zip(runs, seeds)])

    print(f"\n{args.seeds} deployments, 100 nodes, 500x500 m, MADII Table I parameters")
    print(f"LP ceiling (first node death): mean {np.mean(list(ceil.values())):.1f} rounds\n")
    hdr = f"{'policy':<34}{'FND':>7}{'% ceiling':>11}{'worst%':>8}{'HND':>7}{'EE@FND':>9}{'DDV@HND':>10}"
    print(hdr); print("-" * len(hdr))
    for name, r in res.items():
        print(f"{name:<34}{r[0.01]['SR'].mean():>7.1f}{r['pct'].mean():>11.1f}{r['pct'].min():>8.1f}"
              f"{r[0.50]['SR'].mean():>7.1f}{r[0.01]['EE'].mean():>9.0f}{r[0.50]['DDV'].mean():>10.0f}")
    print("-" * len(hdr))
    for name, rows in PAPER.items():
        print(f"{name:<34}{rows[0][0]:>7}{100*(rows[0][0]-1)/np.mean(list(ceil.values())):>11.1f}"
              f"{'-':>8}{rows[3][0]:>7}{rows[0][1]:>9}{rows[3][2]:>10}")
    json.dump({k: {str(b): {m: v.tolist() for m, v in d.items()} if b != "pct" else d.tolist()
                   for b, d in r.items()} for k, r in res.items()},
              open("results_eval.json", "w"), indent=1)
    print("\nwrote results_eval.json")


if __name__ == "__main__":
    main()
