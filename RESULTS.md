# Day-1 results: reproduced MADII-family learner vs classical routing vs the optimum

*Shubham Mittal & Aditya Periwal, NSUT Delhi. Generated 2026-09-25.*

Everything below is measured in **MADII's own simulation world** (Yang et al., IEEE
IoT-J 12(23), Dec 2025): 100 sensors uniform in 500x500 m, sink at the centre, 4 kbit
per sensor per round, first-order radio model Eq. (1)-(2), Table I parameters.

## 1. What is new since the last status

1. **The ceiling is verified, not just computed.** On small networks we enumerated
   *every* acyclic routing tree; the best one never exceeds the LP. On 100-node
   deployments no policy ever exceeds it. (`verify_lp.py`)
2. **Our bench reproduces the paper's FCM baseline for beta >= 10%, and only if
   cluster heads do NOT fuse data.** Without fusion, our FCM lands at 1.00x / 1.47x /
   1.41x the paper's rounds at beta = 10 / 25 / 50%. With fusion it is 4-7x off.
   This settles a modelling question that was open. (`calibrate_fcm.py`)
3. **A trained learner now exists.** We rebuilt MADII's method and trained it, so the
   comparison is no longer against printed numbers alone.

## 2. Classical routing against the provable optimum (30 deployments)

| Policy | rounds to first death | % of ceiling | worst case | rounds to half dead |
|---|---|---|---|---|
| direct to sink | 8.3 | 4.4% | 2.9% | 61.5 |
| FCM clustering (their baseline) | 5.4 | 2.6% | 0.0% | 80.3 |
| min-energy Dijkstra | 42.5 | 24.3% | 14.2% | 159.1 |
| **battery-weighted Dijkstra (one line different)** | **151.4** | **88.4%** | **80.8%** | **162.9** |
| *MADII, as printed in their Table VI* | *19* | *10.6%* | *-* | *89* |

Mean ceiling over the 30 deployments: **169.7 rounds**.

## 3. Reproduced learner

(filled in from the training runs -- see section 5 for the fidelity checks)

## 4. Honest caveats

- The learner is **our reconstruction**, not their code. Reduced model size, and every
  unspecified detail (masking, replay size, episode counts, value factorisation) is
  our choice, marked in the source.
- Their epsilon schedule (1.0 -> 0.1) applies **per sensor**; with 100 sensors acting
  at once this randomises the whole network and every episode dies in one round. We
  measured that, then scaled the schedule to 0.2 -> 0.02.
- Our FCM does **not** reproduce their FCM at first-node-death (5.4 vs 19), though it
  matches for every larger beta. Their FCM row is also identical to MADII's at that
  beta, which is itself odd.
- The ceiling is **first-node-death only**, on a static deployment, in their idealised
  radio model. Nothing here has been through a packet-level simulator yet.

## 5. State at end of 2026-09-25 (session stopped here)

**Done and checked:**
- `env.py` / `policies.py` / `lp_bound.py` / `qnet.py` / `train.py` / `evaluate.py` /
  `experts.py` all written and working. Repo runs end to end.
- LP ceiling verified two ways (`verify_lp.py`): exhaustive enumeration on small
  networks, and no violation by any policy on 100-node deployments.
- Bench validated against the paper's printed baselines, 30 deployments:
  HBA 12.4 vs 14 printed (0.89x), POA 10.9 vs 13 (0.84x), energy efficiency 1.09x and
  1.11x. FCM does not match at first-node-death (5.1 vs 19) though it does for larger
  beta. **Two of three implementable baselines reproduce their numbers.**
- Classical vs ceiling, 30 deployments: battery-weighted Dijkstra 88.4% of optimum
  (worst 80.8%), min-energy 24.3%, direct 4.4%, FCM 2.6%. MADII's printed 19 rounds is
  10.6% of the ceiling.
- Ablation trained and evaluated: transformer DQN with NO imitation reaches first death
  at 8.0 rounds (4.2% of ceiling) -- i.e. no better than sending everything direct.

**Still running overnight:** `madti_v2` (transformer + FCM/HBA/POA imitation) and
`madii_informer` (ProbSparse attention = MADII proper).

**What the learner looks like so far (honest):** performance peaks at the END of the
imitation stage, around 12 rounds to first death -- level with the experts it copies and
close to the paper's printed MADTI (12). The epsilon-greedy stage then DEGRADES it
(training first-death falls to 2-4 rounds). If that holds after the overnight runs, the
finding is that the reproduced learner's ability comes from imitation, not from
reinforcement -- which is consistent with the paper's own ablations, where the
no-imitation variant is the weakest.

**Known issues to fix first tomorrow:**
1. `logs/eval_madti_ckpts.txt` failed: `KeyError: 'ep'` -- the checkpoint's episode
   number lives in the checkpoint dict, not in `args`. One-line fix in that snippet.
2. `logs/*.out` for `madti` and `madte_noil` contain interleaved text from runs that
   were killed and relaunched; read `checkpoints/` and fresh logs instead.
3. Checkpoints saved before the best-model patch keep only the LATEST eval, not the
   best; snapshots in `checkpoints/snapshots/` cover that gap.
