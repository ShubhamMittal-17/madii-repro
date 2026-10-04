# Real traffic: Intel Berkeley Lab (27 test deployments, 15-24 days after 28 Feb 2004)

54 real motes (layout scaled x12 to a 500 m field), send-on-delta traffic (dT 0.5 C, dL 100 lux, +1 heartbeat/hour), E0 = 1.0 J, Holt-Winters alpha 0.05, gamma 0.3 (tuned on training days). L = lifetime to first node death, % of the oracle LP.

| Method | Mean L | Worst | Best | vs battery Dijkstra (wins/ties/losses, p) |
|---|---|---|---|---|
| Battery Dijkstra | 86.0 | 63.0 | 99.6 | – |
| Predictive Dijkstra, Holt-Winters | 84.1 | 51.1 | 99.6 | 4/9/14, p=0.085 |
| Load-aware Dijkstra (ours, no LP) | 90.3 | 68.1 | 99.6 | 19/7/1, p=0.00016 |
| Load-aware Dijkstra, LEST load table (4 tiers) | 90.8 | 77.3 | 99.6 | 16/7/4, p=0.0015 |
| Static LP (solved once) | 54.0 | 16.7 | 81.0 | 0/0/27, p=1.5e-08 |
| Reactive LP coordinator | 85.4 | 51.1 | 99.6 | 13/7/7, p=0.39 |
| Forecast LP v1, Holt-Winters | 93.7 | 68.1 | 99.6 | 20/7/0, p=8.9e-05 |
| Forecast LP v2, Holt-Winters (ours) | 96.9 | 88.8 | 99.6 | 22/5/0, p=4e-05 |
| Forecast LP v2, history mean (no forecast) | 96.6 | 79.9 | 99.6 | 22/5/0, p=4e-05 |
| Forecast LP v2, seasonal-naive | 96.0 | 77.3 | 99.6 | 22/5/0, p=4e-05 |
| Forecast LP v2, perfect forecast (diagnostic) | 96.3 | 79.0 | 99.6 | 22/5/0, p=4e-05 |
| MADII (not retrained) | 11.8 | 0.5 | 31.2 | 0/0/27, p=5.6e-06 |

Ablations:

- v2 Holt-Winters vs v2 history mean (what the forecast adds): +0.4 pts, 6/14/7, p=0.81
- v2 vs v1, Holt-Winters (what the planner adds): +3.2 pts, 11/14/2, p=0.023
- perfect vs Holt-Winters, v2 (room for a better forecast): -0.6 pts, 8/15/4, p=0.69

Forecast error, next-24-hour mean rate per mote (packets/hour, mean absolute error over test starts): holt-winters 0.52, history-mean 1.03, seasonal-naive 0.54, persistence 1.76

Deployments still alive at the end of the data (censored, scored at the data end): Battery Dijkstra 5, Predictive Dijkstra, Holt-Winters 5, Load-aware Dijkstra (ours, no LP) 5, Load-aware Dijkstra, LEST load table (4 tiers) 5, Reactive LP coordinator 6, Forecast LP v1, Holt-Winters 6, Forecast LP v2, Holt-Winters (ours) 6, Forecast LP v2, history mean (no forecast) 6, Forecast LP v2, seasonal-naive 6, Forecast LP v2, perfect forecast (diagnostic) 6
