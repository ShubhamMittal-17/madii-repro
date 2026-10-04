# MADII with the paper's exact exploration schedule (Run A)

Settings: v3 rebuild (Informer, d_model 128) but the paper's epsilon schedule 1.0 -> 0.1. Trained 1000 episodes (300 imitation). 30 held-out deployments.

```
LP ceiling (first node death): mean 169.7 rounds

policy                                FND  % ceiling  worst%    HND   EE@FND   DDV@HND
--------------------------------------------------------------------------------------
direct                                8.3        4.4     2.9   61.5      317     18081
FCM k=10                              5.4        2.6     0.0   80.3      284     25018
min-energy Dijkstra                  42.5       24.3    14.2  159.1     2448     55735
battery Dijkstra a=1                151.4       88.4    80.8  162.9     1618     63843
madii_exact_eps_best (informer, IL)   13.3        7.4     3.9   74.7      441     23273
madii_exact_eps (informer, IL)       14.9        8.2     4.3   64.6      587     20516
--------------------------------------------------------------------------------------
MADII (printed)                        19       10.6       -     89      817     25708
MADTI (printed)                        12        6.5       -     67      352     19692
FCM (printed)                          19       10.6       -     57      555     18704

```

# MADII at the paper's full model size (Run B, partial)

Settings: Informer with d_model 1024, d_ff 4096, 8 heads (27.3M parameters, ~6.5 s per gradient step on 4 CPU threads), paper's low-exploration schedule 0.1 -> 0.01, mixed experts, 1000 episodes planned (300 imitation).
The container restarted at episode 290, so this is the episode-200 checkpoint (imitation stage, ~1,300 gradient steps). train.py has no resume, so a full run means restarting from scratch (~10-12 h).

```
policy                                FND  % ceiling  worst%    HND   EE@FND   DDV@HND
--------------------------------------------------------------------------------------
battery Dijkstra a=1                151.4       88.4    80.8  162.9     1618     63843
madii_papersize_best (informer, IL)   15.4        8.4     1.4   51.0      531     17620
```

# Summary: every MADII configuration vs. battery Dijkstra (30 held-out deployments, % of LP ceiling)

| Configuration | Params | FND (rounds) | % ceiling |
|---|---|---|---|
| v3 rebuild (scaled exploration) | 0.3M | 19.5 | 11.1 |
| Run A: paper's epsilon 1.0 -> 0.1 | 0.3M | 14.9 | 8.2 |
| Run B: paper's model size (ep 200 of 1000) | 27.3M | 15.4 | 8.4 |
| Battery-weighted Dijkstra (no learning) | 0 | 151.4 | 88.4 |

A model 90x larger gives the same ~8-11% as the small one; neither moves toward Dijkstra's 88%.
