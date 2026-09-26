"""Final head-to-head: reproduced MADII vs battery-weighted Dijkstra vs warm-started RL
Dijkstra, all against the provable LP ceiling, on the same 30 held-out deployments
(seeds 0-29, never used for training or checkpoint selection by dijkstra_rl.py).

Paired tests: every arm is compared with battery Dijkstra deployment by deployment
(Wilcoxon signed-rank on rounds to first node death), since deployments differ far more
than policies do.

  .venv/bin/python compare.py                 # writes results_compare.json + results_compare.md
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import env as E, policies as P
import dijkstra_rl as D

BETAS = (0.01, 0.10, 0.25, 0.50)
PAPER = {  # Table VI (SR at beta = 1/10/25/50 %, EE at FND)
    "MADII (as printed in paper)": ((19, 37, 52, 89), 817),
    "MADTI (as printed in paper)": ((12, 17, 28, 67), 352),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--madii", nargs="*", default=None, help="MADII checkpoints (.pt)")
    ap.add_argument("--rl", nargs="*", default=None, help="RL-Dijkstra checkpoints (.npz)")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    ceil = np.array([D.ceiling(s) for s in seeds])

    arms = {"battery Dijkstra (classical)": D.fast_battery_weighted,
            "min-energy Dijkstra": P.min_energy,
            "FCM clustering (paper baseline)": P.fcm_routing}
    for c in args.rl if args.rl is not None else sorted(glob.glob("checkpoints/dijkstra_rl*.npz")):
        arms[f"RL-Dijkstra warm-start [{Path(c).stem}]"] = D.load(c)
    madii = args.madii if args.madii is not None else sorted(glob.glob("checkpoints/*.pt"))
    if madii:
        import torch
        from evaluate import load
        from train import greedy_action
        torch.manual_seed(0)
        for c in madii:
            net, a = load(c)
            expert = "no IL" if not a["il_episodes"] else f"IL from {a.get('experts') or 'fcm'}"
            arms[f"MADII-repro {a['arch']}, {expert} [{Path(c).stem}]"] = \
                (lambda nn_: (lambda w: greedy_action(nn_, w)))(net)

    res = {}
    for name, pol in arms.items():
        runs = [E.run_episode(E.WSN(seed=s), pol, BETAS) for s in seeds]
        res[name] = {"SR": {b: np.array([r[b]["SR"] for r in runs]) for b in BETAS},
                     "EE": np.array([r[0.01]["EE"] for r in runs]),
                     "DDV": np.array([r[0.50]["DDV"] for r in runs])}
        res[name]["pct"] = (res[name]["SR"][0.01] - 1) / ceil * 100
        print(f"done {name}", flush=True)

    ref = res["battery Dijkstra (classical)"]["SR"][0.01]
    lines = [f"{args.seeds} held-out deployments (seeds 0-{args.seeds - 1}), 100 nodes, "
             f"500x500 m, MADII Table I. LP ceiling mean **{ceil.mean():.1f}** rounds.", "",
             "| policy | FND | % of ceiling | worst % | 10% dead | 25% dead | HND | EE@FND | "
             "vs battery Dijkstra (W/T/L, p) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        fnd = r["SR"][0.01]
        d = fnd - ref
        wtl = f"{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}"
        p = "-" if name.startswith("battery") else (
            f"{wilcoxon(fnd, ref).pvalue:.2g}" if np.any(d != 0) else "1")
        lines.append(f"| {name} | {fnd.mean():.1f} | {r['pct'].mean():.1f} | {r['pct'].min():.1f} | "
                     f"{r['SR'][0.10].mean():.1f} | {r['SR'][0.25].mean():.1f} | "
                     f"{r['SR'][0.50].mean():.1f} | {r['EE'].mean():.0f} | "
                     + ("reference" if name.startswith("battery") else f"{wtl}, p={p}") + " |")
    for name, (sr, ee) in PAPER.items():
        lines.append(f"| *{name}* | *{sr[0]}* | *{100 * (sr[0] - 1) / ceil.mean():.1f}* | - | "
                     f"*{sr[1]}* | *{sr[2]}* | *{sr[3]}* | *{ee}* | different bench |")
    table = "\n".join(lines)
    print("\n" + table)
    Path("results_compare.md").write_text(table + "\n")
    json.dump({k: {"FND": v["SR"][0.01].tolist(), "pct_ceiling": v["pct"].tolist(),
                   "SR": {str(b): v["SR"][b].tolist() for b in BETAS},
                   "EE_at_FND": v["EE"].tolist(), "DDV_at_HND": v["DDV"].tolist()}
               for k, v in res.items()} | {"ceiling": ceil.tolist()},
              open("results_compare.json", "w"), indent=1)
    print("\nwrote results_compare.md, results_compare.json")


if __name__ == "__main__":
    main()
