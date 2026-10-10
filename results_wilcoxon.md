# Wilcoxon signed-rank tests

### Time-varying traffic, R+N+B+D pooled (120 deployments), against LEST load-table Dijkstra

Each row: method minus LEST load-table Dijkstra, paired by deployment.

| Method | n | Mean diff | Median diff | W / T / L | Wilcoxon p | Holm p | r |
|---|---|---|---|---|---|---|---|
| LP planner v2 | 120 | +6.3 | +5.6 | 119 / 0 / 1 | 2.1e-21 | 2.2e-20 | +1.00 |
| LP planner v2, historical mean | 120 | +5.7 | +5.2 | 116 / 3 / 1 | 6.8e-21 | 3.4e-20 | +1.00 |
| Load-aware Dijkstra, exact load | 120 | +0.5 | +0.5 | 61 / 29 / 30 | 6.4e-04 | 6.4e-04 | +0.41 |
| Battery Dijkstra | 120 | -5.9 | -4.6 | 1 / 1 / 118 | 3.5e-21 | 2.5e-20 | -1.00 |
| Price-guided Dijkstra | 120 | -4.9 | -3.5 | 6 / 5 / 109 | 3.8e-17 | 1.1e-16 | -0.90 |
| Predictive Dijkstra | 120 | -6.6 | -5.2 | 0 / 2 / 118 | 4.2e-21 | 2.5e-20 | -1.00 |
| Load table only (no battery) | 120 | -45.0 | -45.6 | 0 / 0 / 120 | 2.0e-21 | 2.2e-20 | -1.00 |
| Static LP | 120 | -23.9 | -25.1 | 7 / 1 / 112 | 1.0e-20 | 4.1e-20 | -0.99 |
| Reactive LP | 120 | -27.2 | -26.7 | 31 / 0 / 89 | 7.3e-16 | 1.5e-15 | -0.85 |
| EBR-DA | 120 | -75.2 | -75.6 | 0 / 0 / 120 | 2.0e-21 | 2.2e-20 | -1.00 |
| MADII | 120 | -76.0 | -77.1 | 0 / 0 / 120 | 2.0e-21 | 2.2e-20 | -1.00 |

### Time-varying traffic, R+N+B+D pooled (120 deployments), against Battery Dijkstra

Each row: method minus Battery Dijkstra, paired by deployment.

| Method | n | Mean diff | Median diff | W / T / L | Wilcoxon p | Holm p | r |
|---|---|---|---|---|---|---|---|
| LP planner v2 | 120 | +12.2 | +12.0 | 120 / 0 / 0 | 2.0e-21 | 2.2e-20 | +1.00 |
| LEST load-table Dijkstra | 120 | +5.9 | +4.6 | 118 / 1 / 1 | 3.5e-21 | 2.2e-20 | +1.00 |
| LP planner v2, historical mean | 120 | +11.6 | +12.0 | 120 / 0 / 0 | 2.0e-21 | 2.2e-20 | +1.00 |
| Load-aware Dijkstra, exact load | 120 | +6.3 | +5.4 | 116 / 3 / 1 | 7.2e-21 | 3.6e-20 | +1.00 |
| Price-guided Dijkstra | 120 | +1.0 | +0.6 | 66 / 22 / 32 | 2.2e-04 | 4.5e-04 | +0.43 |
| Predictive Dijkstra | 120 | -0.8 | +0.0 | 24 / 47 / 49 | 0.007 | 0.007 | -0.36 |
| Load table only (no battery) | 120 | -39.1 | -38.8 | 0 / 0 / 120 | 2.0e-21 | 2.2e-20 | -1.00 |
| Static LP | 120 | -18.0 | -18.7 | 16 / 1 / 103 | 1.4e-18 | 5.7e-18 | -0.93 |
| Reactive LP | 120 | -21.3 | -19.4 | 32 / 1 / 87 | 6.5e-14 | 1.9e-13 | -0.79 |
| EBR-DA | 120 | -69.4 | -69.9 | 0 / 0 / 120 | 2.0e-21 | 2.2e-20 | -1.00 |
| MADII | 120 | -70.1 | -70.3 | 0 / 0 / 120 | 2.0e-21 | 2.2e-20 | -1.00 |

### Per scenario (30 deployments each)

Each row: method minus the second method, paired by deployment.

| Method | n | Mean diff | Median diff | W / T / L | Wilcoxon p | Holm p | r |
|---|---|---|---|---|---|---|---|
| R: LEST load-table Dijkstra vs Battery Dijkstra | 30 | +6.3 | +6.0 | 29 / 1 / 0 | 2.6e-06 | 1.4e-05 | +1.00 |
| R: LP planner v2 vs LEST load-table Dijkstra | 30 | +6.9 | +5.6 | 30 / 0 / 0 | 1.9e-09 | 3.0e-08 | +1.00 |
| R: LP planner v2 vs Battery Dijkstra | 30 | +13.2 | +12.7 | 30 / 0 / 0 | 1.9e-09 | 3.0e-08 | +1.00 |
| R: LP planner v2 vs LP planner v2, historical mean | 30 | +0.7 | +0.6 | 17 / 6 / 7 | 0.047 | 0.188 | +0.46 |
| N: LEST load-table Dijkstra vs Battery Dijkstra | 30 | +4.0 | +3.8 | 30 / 0 / 0 | 1.9e-09 | 3.0e-08 | +1.00 |
| N: LP planner v2 vs LEST load-table Dijkstra | 30 | +5.9 | +6.3 | 30 / 0 / 0 | 1.7e-06 | 1.4e-05 | +1.00 |
| N: LP planner v2 vs Battery Dijkstra | 30 | +9.9 | +9.2 | 30 / 0 / 0 | 1.9e-09 | 3.0e-08 | +1.00 |
| N: LP planner v2 vs LP planner v2, historical mean | 30 | +0.0 | +0.0 | 0 / 30 / 0 | 1 | 1 | +0.00 |
| B: LEST load-table Dijkstra vs Battery Dijkstra | 30 | +6.2 | +4.7 | 30 / 0 / 0 | 1.7e-06 | 1.4e-05 | +1.00 |
| B: LP planner v2 vs LEST load-table Dijkstra | 30 | +6.8 | +5.7 | 30 / 0 / 0 | 1.9e-09 | 3.0e-08 | +1.00 |
| B: LP planner v2 vs Battery Dijkstra | 30 | +13.1 | +13.7 | 30 / 0 / 0 | 1.7e-06 | 1.4e-05 | +1.00 |
| B: LP planner v2 vs LP planner v2, historical mean | 30 | +1.0 | +0.0 | 14 / 6 / 10 | 0.136 | 0.407 | +0.35 |
| D: LEST load-table Dijkstra vs Battery Dijkstra | 30 | +7.0 | +6.9 | 29 / 0 / 1 | 5.6e-09 | 5.0e-08 | +0.99 |
| D: LP planner v2 vs LEST load-table Dijkstra | 30 | +5.5 | +4.5 | 29 / 0 / 1 | 3.7e-09 | 3.7e-08 | +1.00 |
| D: LP planner v2 vs Battery Dijkstra | 30 | +12.6 | +12.5 | 30 / 0 / 0 | 1.9e-09 | 3.0e-08 | +1.00 |
| D: LP planner v2 vs LP planner v2, historical mean | 30 | +0.5 | +0.0 | 13 / 8 / 9 | 0.235 | 0.470 | +0.30 |

### Intel Lab traffic, E0 = 1.0 J (27 deployments), against LEST load-table Dijkstra

Each row: method minus LEST load-table Dijkstra, paired by deployment.

| Method | n | Mean diff | Median diff | W / T / L | Wilcoxon p | Holm p | r |
|---|---|---|---|---|---|---|---|
| Battery Dijkstra | 27 | -4.7 | -0.5 | 4 / 7 / 16 | 0.002 | 0.006 | -0.81 |
| Predictive Dijkstra | 27 | -6.7 | -1.5 | 0 / 8 / 19 | 3.8e-06 | 2.3e-05 | -1.00 |
| Load-aware Dijkstra, exact load | 27 | -0.5 | +0.0 | 13 / 10 / 4 | 0.089 | 0.177 | +0.48 |
| Static LP | 27 | -36.8 | -31.9 | 0 / 0 / 27 | 1.5e-08 | 1.6e-07 | -1.00 |
| Reactive LP | 27 | -5.4 | +0.0 | 9 / 6 / 12 | 0.229 | 0.229 | -0.31 |
| LP planner v1 | 27 | +3.0 | +2.1 | 17 / 7 / 3 | 0.002 | 0.006 | +0.75 |
| LP planner v2 | 27 | +6.2 | +6.5 | 21 / 6 / 0 | 9.5e-07 | 7.6e-06 | +1.00 |
| LP planner v2, historical mean | 27 | +5.8 | +6.5 | 20 / 7 / 0 | 1.9e-06 | 1.3e-05 | +1.00 |
| LP planner v2, perfect forecast | 27 | +5.5 | +6.9 | 21 / 5 / 1 | 1.8e-04 | 8.8e-04 | +0.84 |
| MADII | 27 | -78.9 | -76.6 | 0 / 0 / 27 | 1.5e-08 | 1.6e-07 | -1.00 |
| EBR-DA | 27 | -73.7 | -72.8 | 0 / 0 / 27 | 1.5e-08 | 1.6e-07 | -1.00 |

### Intel Lab traffic, E0 = 1.0 J (27 deployments), against Battery Dijkstra

Each row: method minus Battery Dijkstra, paired by deployment.

| Method | n | Mean diff | Median diff | W / T / L | Wilcoxon p | Holm p | r |
|---|---|---|---|---|---|---|---|
| Predictive Dijkstra | 27 | -1.9 | -0.4 | 4 / 9 / 14 | 0.090 | 0.180 | -0.46 |
| Load-aware Dijkstra, exact load | 27 | +4.3 | +1.4 | 19 / 7 / 1 | 1.3e-05 | 5.3e-05 | +0.96 |
| LEST load-table Dijkstra | 27 | +4.7 | +0.5 | 16 / 7 / 4 | 0.002 | 0.005 | +0.81 |
| Static LP | 27 | -32.0 | -26.0 | 0 / 0 / 27 | 1.5e-08 | 1.6e-07 | -1.00 |
| Reactive LP | 27 | -0.6 | +0.0 | 13 / 7 / 7 | 0.409 | 0.409 | +0.22 |
| LP planner v1 | 27 | +7.7 | +6.5 | 20 / 7 / 0 | 1.9e-06 | 1.1e-05 | +1.00 |
| LP planner v2 | 27 | +10.9 | +9.1 | 22 / 5 / 0 | 4.8e-07 | 4.3e-06 | +1.00 |
| LP planner v2, historical mean | 27 | +10.6 | +8.9 | 22 / 5 / 0 | 4.8e-07 | 4.3e-06 | +1.00 |
| LP planner v2, perfect forecast | 27 | +10.3 | +7.8 | 22 / 5 / 0 | 4.8e-07 | 4.3e-06 | +1.00 |
| MADII | 27 | -74.2 | -75.0 | 0 / 0 / 27 | 5.6e-06 | 2.8e-05 | -1.00 |
| EBR-DA | 27 | -68.9 | -68.2 | 0 / 0 / 27 | 1.5e-08 | 1.6e-07 | -1.00 |

### Static traffic, MADII's setting (30 deployments), against battery Dijkstra

Each row: method minus battery Dijkstra, paired by deployment.

| Method | n | Mean diff | Median diff | W / T / L | Wilcoxon p | Holm p | r |
|---|---|---|---|---|---|---|---|
| LP-flow adaptive | 30 | +6.1 | +6.4 | 29 / 0 / 1 | 2.0e-07 | 6.1e-07 | +0.94 |
| RL warm-started from Dijkstra | 30 | +0.5 | +0.3 | 15 / 5 / 10 | 0.241 | 0.482 | +0.27 |
| LP-flow static | 30 | -0.8 | +1.1 | 16 / 0 / 14 | 0.824 | 0.824 | +0.05 |
| Min-energy Dijkstra | 30 | -64.1 | -63.7 | 0 / 0 / 30 | 1.9e-09 | 1.1e-08 | -1.00 |
| MADII | 30 | -77.3 | -77.2 | 0 / 0 / 30 | 1.9e-09 | 1.1e-08 | -1.00 |
| FCM clustering | 30 | -85.7 | -85.4 | 0 / 0 / 30 | 1.9e-09 | 1.1e-08 | -1.00 |

r is the matched-pairs rank-biserial correlation (+1: the method is better on every deployment).
Ties (|difference| < 1e-9) are counted in T and dropped from the test. Holm adjusts within each table.
