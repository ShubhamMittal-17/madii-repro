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
