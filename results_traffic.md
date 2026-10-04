# Traffic metrics in MADII's setting (30 held-out deployments, constant traffic)

| Router | FND round | HND round | Data by FND (kbit) | Data by HND (kbit) | kbit per J (to HND) | Delivery ratio | Hops per packet |
|---|---|---|---|---|---|---|---|
| Battery Dijkstra | 151.4 | 162.9 | 60555 | 63843 | 1286 | 98.85% | 3.53 |
| Min-energy Dijkstra | 42.5 | 159.1 | 16982 | 55735 | 1336 | 99.31% | 3.96 |
| Load-aware Dijkstra (ours, no LP) | 158.5 | 164.4 | 63393 | 64880 | 1310 | 99.11% | 3.34 |
| Planner v1 (LP every 4 rounds) | 165.9 | 170.1 | 66334 | 67294 | 1350 | 99.20% | 3.22 |
| Planner v2 (LP every round, ours) | 167.7 | 170.7 | 67040 | 67475 | 1355 | 99.07% | 3.23 |
| MADII (v3 rebuild) | 27.5 | 73.2 | 10982 | 25359 | 681 | 99.20% | 2.61 |

MADII as printed in its paper (same setting): FND 19, HND 89, EE@FND 817, DDV@HND 25708.
