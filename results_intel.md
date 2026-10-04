# Real traffic: Intel Berkeley Lab (27 test deployments, 15-24 days after 28 Feb 2004)

54 real motes (layout scaled x12 to a 500 m field), send-on-delta traffic (dT 0.5 C, dL 100 lux, +1 heartbeat/hour), E0 = 1.5 J, Holt-Winters alpha 0.05, gamma 0.3 (tuned on training days). L = lifetime to first node death, % of the oracle LP.

| Method | Mean L | Worst | Best | vs battery Dijkstra (wins/ties/losses, p) |
|---|---|---|---|---|
| Battery Dijkstra | 93.3 | 69.7 | 99.7 | – |
| Predictive Dijkstra, Holt-Winters | 89.3 | 55.3 | 99.7 | 4/14/9, p=0.033 |
| Load-aware Dijkstra (ours, no LP) | 94.7 | 76.3 | 99.7 | 11/14/2, p=0.033 |
| Static LP (solved once) | 62.8 | 16.7 | 99.6 | 0/5/22, p=4e-05 |
| Reactive LP coordinator | 94.2 | 48.3 | 99.7 | 11/14/2, p=0.028 |
| Forecast LP v1, Holt-Winters | 96.7 | 79.5 | 99.7 | 13/14/0, p=0.0015 |
| Forecast LP v2, Holt-Winters (ours) | 98.0 | 86.7 | 99.7 | 14/13/0, p=0.00098 |
| Forecast LP v2, history mean (no forecast) | 98.3 | 86.7 | 99.7 | 14/13/0, p=0.00098 |
| Forecast LP v2, seasonal-naive | 98.3 | 86.7 | 99.7 | 14/13/0, p=0.00098 |
| Forecast LP v2, perfect forecast (diagnostic) | 98.3 | 87.7 | 99.7 | 14/13/0, p=0.00098 |
| MADII (not retrained) | 17.1 | 6.4 | 51.0 | 0/0/27, p=1.5e-08 |

Ablations:

- v2 Holt-Winters vs v2 history mean (what the forecast adds): -0.3 pts, 0/18/9, p=0.0077
- v2 vs v1, Holt-Winters (what the planner adds): +1.3 pts, 8/16/3, p=0.041
- perfect vs Holt-Winters, v2 (room for a better forecast): +0.3 pts, 7/19/1, p=0.017

Forecast error, next-24-hour mean rate per mote (packets/hour, mean absolute error over test starts): holt-winters 0.52, history-mean 1.03, seasonal-naive 0.54, persistence 1.76

Deployments still alive at the end of the data (censored, scored at the data end): Battery Dijkstra 13, Predictive Dijkstra, Holt-Winters 13, Load-aware Dijkstra (ours, no LP) 14, Static LP (solved once) 5, Reactive LP coordinator 14, Forecast LP v1, Holt-Winters 15, Forecast LP v2, Holt-Winters (ours) 15, Forecast LP v2, history mean (no forecast) 15, Forecast LP v2, seasonal-naive 14, Forecast LP v2, perfect forecast (diagnostic) 15
