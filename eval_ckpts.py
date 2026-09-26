"""Evaluate saved checkpoints on the same 30 deployments the classical policies use."""
import glob, sys
import numpy as np
import env as E
from evaluate import load
from train import greedy_action
from lp_bound import max_lifetime_T

seeds = range(30)
ceil = np.mean([max_lifetime_T(E.WSN(seed=s)) for s in seeds])
files = sys.argv[1:] or sorted(glob.glob("checkpoints/*.pt")) + sorted(glob.glob("checkpoints/snapshots/*.pt"))
print(f"ceiling {ceil:.1f} rounds\n{'checkpoint':<40}{'ep':>5}{'FND':>7}{'%ceil':>7}{'HND':>7}{'EE@FND':>8}")
for c in files:
    net, a = load(c)
    import torch
    ep = torch.load(c, map_location="cpu", weights_only=False)["ep"]
    rows = [E.run_episode(E.WSN(seed=s), lambda w: greedy_action(net, w)) for s in seeds]
    f = lambda b, k: np.mean([r[b][k] for r in rows])
    print(f"{c.replace('checkpoints/',''):<40}{ep:>5}{f(0.01,'SR'):>7.1f}"
          f"{100*(f(0.01,'SR')-1)/ceil:>7.1f}{f(0.50,'SR'):>7.1f}{f(0.01,'EE'):>8.0f}", flush=True)
