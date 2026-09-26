"""Does our bench reproduce the paper's FCM row? Their Table VI: FND 19, EE 555, DDV 7596.

FCM is the one baseline we can implement exactly, so it tells us whether our simulator
matches theirs. The open question is whether their cluster heads FUSE the data they
collect; the MADII text forbids fusion for MADII itself but says nothing about the
baselines, and it changes the answer by an order of magnitude.
"""
import numpy as np
import env as E, policies as P

PAPER = {0.01: (19, 555, 7596), 0.10: (27, 533, 10636), 0.25: (32, 531, 12228), 0.50: (57, 597, 18704)}
SEEDS = range(30)

for agg in (False, True):
    rows = [E.run_episode(E.WSN(seed=s, aggregate=agg), P.fcm_routing) for s in SEEDS]
    print(f"\nfusion at cluster head: {agg}")
    print(f"{'beta':>6}{'our SR':>8}{'paper':>7}{'ratio':>7}{'our EE':>9}{'paper':>7}{'ratio':>7}")
    for b, (sr, ee, ddv) in PAPER.items():
        o_sr = np.mean([r[b]["SR"] for r in rows]); o_ee = np.mean([r[b]["EE"] for r in rows])
        print(f"{b:>6}{o_sr:>8.1f}{sr:>7}{o_sr/sr:>7.2f}{o_ee:>9.0f}{ee:>7}{o_ee/ee:>7.2f}")
