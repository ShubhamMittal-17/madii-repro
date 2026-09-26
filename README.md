# MADII reproduction + classical routing + the provable ceiling

Reproduction of **MADII** (Yang et al., "An Energy-Efficient and Transmission-Efficient
Adaptive Routing Algorithm Using Deep Reinforcement Learning for WSNs", IEEE Internet of
Things Journal 12(23), Dec 2025) and a head-to-head against classical routing and
against the maximum-lifetime optimum.

The paper releases no code, so this is a best-effort rebuild from the text. Every
choice the paper leaves open is marked in the source (`env.py` R1-R4, `train.py` C1-C6).

## Files
| File | What |
|---|---|
| `env.py` | The paper's world: Eq. (1)-(2) radio model, Table I parameters, round mechanics, reward Eq. (8), metrics Eq. (19)-(22) |
| `policies.py` | `direct`, `min_energy` (per-round energy optimum), `battery_weighted` (our coordinator), `fcm_routing` (their FCM baseline), `random_routing` |
| `lp_bound.py` | Chang-Tassiulas maximum-lifetime LP = the provable ceiling on first-node-death |
| `verify_lp.py` | Checks the ceiling by exhaustive enumeration on tiny networks, and that no policy ever exceeds it |
| `qnet.py` | Q-network producing their Fig. 3 table of Q values, N x (N+1). `arch=transformer` = their MADTI ablation; `arch=informer` = ProbSparse attention (MADII proper) |
| `train.py` | DQN per their Algorithm 1: imitation stage from FCM, then epsilon-greedy |
| `evaluate.py` | Head-to-head table vs classical policies, the ceiling, and their printed Table VI |
| `calibrate_fcm.py` | Does our bench reproduce their FCM row? |

## Setup
```
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy scipy
```

## Run
```
.venv/bin/python verify_lp.py                      # ceiling sanity checks
.venv/bin/python calibrate_fcm.py                  # our bench vs their FCM row
.venv/bin/python train.py --arch transformer --tag madti --episodes 900 --il-episodes 200
.venv/bin/python evaluate.py --ckpt checkpoints/madti.pt --seeds 30
```
