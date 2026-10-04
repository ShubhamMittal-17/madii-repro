# Does prediction sense the pattern and act on it?

30 held-out deployments; hotspot = 20% of nodes. Shares are the hotspot nodes' share of all relayed packets / energy spent.

## Daily surge

Sensing: top-20% forecast nodes before the surge are true hotspot nodes 100% of the time; forecast peak is off by 0.14 hours on average.

| Router | Rounds to FND | Hotspot relay share: pre-surge | surge | rest | Hotspot energy share: pre | surge | rest | First death is a hotspot node | Route churn / round |
|---|---|---|---|---|---|---|---|---|---|
| Battery Dijkstra | 134.2 | 30.2% | 43.6% | 28.6% | 22.7% | 43.7% | 21.5% | 47% | 20.9% |
| Predictive Dijkstra (HW) | 131.3 | 28.2% | 46.8% | 27.9% | 21.5% | 45.6% | 21.1% | 67% | 20.9% |
| Forecast LP (HW, ours) | 144.3 | 27.4% | 42.4% | 26.0% | 14.6% | 37.9% | 12.9% | 7% | 33.2% |

## Surge + bursts

Sensing: top-20% forecast nodes before the surge are true hotspot nodes 100% of the time; forecast peak is off by 0.16 hours on average.

| Router | Rounds to FND | Hotspot relay share: pre-surge | surge | rest | Hotspot energy share: pre | surge | rest | First death is a hotspot node | Route churn / round |
|---|---|---|---|---|---|---|---|---|---|
| Battery Dijkstra | 132.9 | 30.1% | 43.3% | 28.8% | 22.5% | 43.4% | 21.8% | 57% | 20.9% |
| Predictive Dijkstra (HW) | 131.8 | 28.1% | 46.7% | 27.9% | 21.6% | 45.5% | 21.2% | 70% | 21.0% |
| Forecast LP (HW, ours) | 141.2 | 27.6% | 41.9% | 26.1% | 14.8% | 37.6% | 12.9% | 13% | 33.1% |
