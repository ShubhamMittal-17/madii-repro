"""Does our bench reproduce the paper's printed baseline rows?

We can implement three of their seven baselines exactly as described: FCM (Table I says
10 clusters) and the two metaheuristics whose fitness the paper gives (Eq. 24, a = 0.54):
HBA and POA. If our numbers match theirs, our simulator is their simulator.
"""
import numpy as np
import env as E, policies as P, experts as X

PRINTED = {  # Table VI: rounds to first node death, and EE at that point
    "FCM": (19, 555), "HBA": (14, 391), "POA": (13, 379),
}
SEEDS = range(30)
rng = np.random.default_rng(0)
arms = {"FCM": lambda w: P.fcm_routing(w, rng=rng),
        "HBA": lambda w: X.hba_routing(w, rng=rng),
        "POA": lambda w: X.poa_routing(w, rng=rng)}

print(f"{'baseline':<8}{'our FND':>9}{'printed':>9}{'ratio':>7}{'our EE':>9}{'printed':>9}{'ratio':>7}")
for name, pol in arms.items():
    rows = [E.run_episode(E.WSN(seed=s), pol) for s in SEEDS]
    sr = np.mean([r[0.01]["SR"] for r in rows]); ee = np.mean([r[0.01]["EE"] for r in rows])
    p_sr, p_ee = PRINTED[name]
    print(f"{name:<8}{sr:>9.1f}{p_sr:>9}{sr/p_sr:>7.2f}{ee:>9.0f}{p_ee:>9}{ee/p_ee:>7.2f}")
