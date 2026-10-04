# Tuning the forecast-LP planner (validation deployments only)

Everything here was chosen on validation deployments 500-511 (12 per scenario) with the
scenarios R (daily surge), N (constant) and B (surge + bursts). The 30 test deployments
(0-29) were not used until the chosen configuration was fixed. Score: lifetime to first
node death, % of the oracle LP. Cost: LP solves per round at the sink.
Reproduce: `.venv/bin/python tune_forecast_lp.py` (configs in the script and logs/tune_round*_extra.json).

| Planner configuration (Holt-Winters unless stated) | Surge | Constant | Burst | Mean | Worst | LP solves / round |
|---|---|---|---|---|---|---|
| Published planner (every 4 rounds, round split) | 88.4 | 97.4 | 88.8 | 91.5 | 76.5 | 0.25 |
| + min-energy tie-break (mu 1e-3) | 89.5 | 97.4 | 89.8 | 92.2 | 76.5 | 0.25 |
| + min-energy tie-break (mu 1e-2) | 89.5 | 97.4 | 89.8 | 92.2 | 76.5 | 0.25 |
| + bit-weighted split (fixed) | 89.8 | 97.1 | 89.4 | 92.1 | 82.7 | 0.25 |
| + bit split + tie-break | 91.4 | 97.0 | 90.0 | 92.8 | 82.7 | 0.25 |
| horizon 12 rounds | 88.7 | 97.4 | 87.6 | 91.2 | 79.6 | 0.25 |
| horizon 48 rounds | 89.0 | 97.4 | 88.8 | 91.7 | 76.5 | 0.25 |
| horizon = remaining lifetime | 89.1 | 97.4 | 89.0 | 91.8 | 77.2 | 0.25 |
| bit split + tie-break + lifetime horizon | 89.8 | 97.0 | 89.9 | 92.2 | 82.7 | 0.25 |
| re-solve every 8 rounds | 83.5 | 95.7 | 81.4 | 86.9 | 66.2 | 0.13 |
| re-solve every 2 rounds | 93.1 | 98.0 | 91.2 | 94.1 | 80.1 | 0.50 |
| every 2 + bit split + tie-break | 91.8 | 98.0 | 92.4 | 94.1 | 77.2 | 0.50 |
| every 2 + bit split + tie-break + lifetime horizon | 93.6 | 98.0 | 92.1 | 94.6 | 85.8 | 0.50 |
| re-solve every round | 95.9 | 98.5 | 95.8 | 96.8 | 93.2 | 1.00 |
| **re-solve every round + tie-break (chosen)** | 95.9 | 98.5 | 96.2 | 96.9 | 93.2 | 1.00 |
| every round + tie-break + lifetime horizon | 96.0 | 98.5 | 94.8 | 96.4 | 90.4 | 1.00 |
| every round + tie-break + carry credits | 85.8 | 92.2 | 65.4 | 81.1 | 24.0 | 1.00 |
| burst-triggered re-solve (3 MAE) | 95.1 | 97.4 | 96.0 | 96.1 | 91.2 | 0.69 |
| burst-triggered re-solve (2 MAE) | 95.9 | 97.4 | 95.8 | 96.4 | 93.2 | 0.75 |
| perfect forecast, published planner | 88.3 | 97.4 | 91.0 | 92.2 | 79.2 | 0.25 |
| perfect forecast, bit split + tie-break | 92.6 | 97.0 | 91.3 | 93.6 | 85.8 | 0.25 |
| perfect forecast, every round + tie-break | 95.8 | 98.5 | 96.1 | 96.8 | 90.4 | 1.00 |
| historical mean (no forecast), published planner | 89.3 | 97.4 | 86.9 | 91.2 | 68.5 | 0.25 |
| historical mean, bit split + tie-break | 90.6 | 97.0 | 88.4 | 92.0 | 66.7 | 0.25 |
| historical mean, every 2 + bit split + tie-break | 94.9 | 98.0 | 94.6 | 95.8 | 92.0 | 0.50 |
| historical mean, every round | 95.0 | 98.5 | 95.6 | 96.4 | 89.3 | 1.00 |
| historical mean, every round + tie-break | 95.0 | 98.5 | 95.8 | 96.4 | 89.3 | 1.00 |
| seasonal-naive, bit split + tie-break | 90.4 | 97.0 | 83.3 | 90.3 | 31.5 | 0.25 |

## What each change did

1. **Re-solving often matters most.** Every 8 rounds 86.9, every 4 (published) 91.5,
   every 2 94.1, every round 96.8. Battery Dijkstra already re-plans every round, so every
   round is the like-for-like setting. In this energy model every round already carries a
   route advertisement that every node receives, so a re-solve costs the nodes no extra
   energy. It costs about 0.2 s of LP time at the sink per round for 100 nodes.
2. **Min-energy tie-break.** Maximising lifetime alone leaves the flows of non-bottleneck
   nodes arbitrary, so some routes waste energy on long links. Adding a tiny total-energy cost
   gives +0.7 at every 4 rounds and +0.1 at every round. It is kept because it never hurt.
3. **Bit-weighted split.** +0.5 at every 4 rounds, but no effect at every round, where each
   node always uses the LP's largest-share hop. The first version scored 21.9% because of a
   bug: it picked hops on the bare deficit, and credits reset every 4 rounds. So a 9%-share
   hop 227 m away got 1 round in 4, and long multipath links drained nodes 11x faster. Fixed
   (smooth deficit round-robin); not needed in the chosen setting.
4. **Carrying credits across re-solves is harmful** (81.1 at every round). Each re-solve
   starts from the current batteries, so it already corrects past deviations. Carrying the
   credits corrects them twice.
5. **Forecast horizon.** 12, 24, 48 rounds or the remaining lifetime all land within +-0.3 at
   every 4 rounds; the lifetime horizon is -0.4 at every round. Kept at one day (24).
6. **Burst-triggered re-solve** is not selective. With 100 nodes and Poisson traffic, some
   node crosses a 2-3 MAE band in 69-75% of rounds, so it works out as frequent re-solving
   (96.1-96.4 at 0.69-0.75 solves/round), not as a separate mechanism.
7. **What the forecast adds.** Compared with a fixed historical rate fed to the same planner:
   - every 4 rounds: 91.5 vs 91.2
   - bit split + tie-break: 92.8 vs 92.0
   - every 2 rounds: 94.1 vs 95.8 (the historical rate ahead)
   - every round: 96.9 vs 96.4
   - a perfect forecast at every round: 96.8

   On this stationary synthetic traffic the forecast adds about half a point at most. The
   planner choices add about five.
8. **Seasonal-naive is fragile.** With bit split + tie-break it scores 90.3, with a worst
   deployment of 31.5%: it copies yesterday's bursts into today's plan.

**Chosen (Forecast LP v2):** re-solve every round, min-energy tie-break (mu = 1e-3).
Holt-Winters alpha 0.05, beta 0, gamma 0.1, one-day horizon, round split.
