"""Sanity checks on the max-lifetime LP (the ceiling).

1. On a tiny network, enumerate EVERY acyclic routing tree, simulate each to first
   node death, and check the best static tree does not exceed the ceiling.
   CONVENTION: SR counts the round a node dies in, so a node that completes T* rounds
   of delivery dies during round floor(T*)+1. The valid test is SR <= floor(T*) + 1.
2. On full 100-node deployments, check no policy we have ever exceeds the ceiling.
"""
import itertools
import numpy as np
import env as E, policies as P
from lp_bound import max_lifetime_T


def acyclic(parent, n):
    for i in range(n):
        seen, u = set(), i
        while u != n:
            if u in seen:
                return False
            seen.add(u); u = parent[u]
    return True


def best_static_tree(seed, n):
    best, best_p = 0, None
    for combo in itertools.product(range(n + 1), repeat=n):
        if any(combo[i] == i for i in range(n)) or not acyclic(combo, n):
            continue
        w = E.WSN(n=n, seed=seed)
        r = E.run_episode(w, lambda _w, c=combo: np.array(c), betas=(0.01,), max_rounds=3000)
        if r[0.01]["SR"] > best:
            best, best_p = r[0.01]["SR"], combo
    return best, best_p


print("check 1: tiny networks, exhaustive over all routing trees")
print(f"{'n':>3}{'seed':>5}{'best static tree':>18}{'LP ceiling':>12}{'LP >= tree?':>13}")
for n in (4, 5):
    for seed in (0, 1, 2):
        tree, _ = best_static_tree(seed, n)
        lp = max_lifetime_T(E.WSN(n=n, seed=seed))
        ok = tree <= np.floor(lp) + 1 + 1e-9
        print(f"{n:>3}{seed:>5}{tree:>18}{lp:>12.1f}{str(bool(ok)):>13}")

print("\ncheck 2: 100 nodes, does any policy ever beat the ceiling?")
viol = 0
for seed in range(10):
    lp = max_lifetime_T(E.WSN(seed=seed))
    for name, pol in (("direct", P.direct), ("min-energy", P.min_energy),
                      ("battery a=1", P.battery_weighted), ("FCM", P.fcm_routing)):
        sr = E.run_episode(E.WSN(seed=seed), pol, betas=(0.01,))[0.01]["SR"]
        if sr > np.floor(lp) + 1 + 1e-9:
            print(f"  VIOLATION seed {seed} {name}: {sr} > {lp:.1f}")
            viol += 1
print(f"  violations: {viol} (expected 0)")
