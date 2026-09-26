30 held-out deployments (seeds 0-29), 100 nodes, 500x500 m, MADII Table I. LP ceiling mean **169.7** rounds.

| policy | FND | % of ceiling | worst % | 10% dead | 25% dead | HND | EE@FND | vs battery Dijkstra (W/T/L, p) |
|---|---|---|---|---|---|---|---|---|
| battery Dijkstra (classical) | 151.4 | 88.4 | 80.8 | 157.5 | 160.8 | 162.9 | 1618 | reference |
| min-energy Dijkstra | 42.5 | 24.3 | 14.2 | 90.6 | 130.1 | 159.1 | 2448 | 0/0/30, p=1.7e-06 |
| FCM clustering (paper baseline) | 5.4 | 2.6 | 0.0 | 27.1 | 46.9 | 80.3 | 284 | 0/0/30, p=1.7e-06 |
| LP-flow static (classical) | 149.6 | 87.5 | 53.2 | 163.5 | 168.2 | 170.3 | 1356 | 16/0/14, p=0.89 |
| LP-flow adaptive (classical) | 161.5 | 94.5 | 81.0 | 167.7 | 168.6 | 169.4 | 1360 | 29/0/1, p=6.4e-06 |
| RL-Dijkstra warm-start [dijkstra_rl] | 151.1 | 88.2 | 82.4 | 157.6 | 160.8 | 162.9 | 1621 | 11/8/11, p=0.83 |
| RL-Dijkstra warm-start [dijkstra_rl_v2] | 152.0 | 88.9 | 83.0 | 158.5 | 161.1 | 163.1 | 1612 | 15/5/10, p=0.31 |
| MADII-repro transformer, IL from fcm [madti] | 6.0 | 3.0 | 1.8 | 13.1 | 24.3 | 56.5 | 236 | 0/0/30, p=1.7e-06 |
| MADII-repro transformer, IL from mixed [madti_v2] | 8.3 | 4.4 | 2.9 | 16.1 | 27.4 | 61.5 | 317 | 0/0/30, p=1.7e-06 |
| MADII-repro transformer, no IL [madte_noil] | 8.0 | 4.2 | 2.9 | 12.9 | 18.5 | 27.8 | 284 | 0/0/30, p=1.7e-06 |
| MADII-repro informer, IL from mixed [madii_informer_best] | 5.3 | 2.6 | 1.2 | 13.8 | 25.0 | 57.8 | 212 | 0/0/30, p=1.7e-06 |
| *MADII (as printed in paper)* | *19* | *10.6* | - | *37* | *52* | *89* | *817* | different bench |
| *MADTI (as printed in paper)* | *12* | *6.5* | - | *17* | *28* | *67* | *352* | different bench |
