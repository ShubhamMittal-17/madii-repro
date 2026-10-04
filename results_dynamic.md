30 held-out deployments per scenario; L = lifetime to first node death, % of the oracle LP.
S = equal-weight mean over the three scored scenarios; Pattern shift (robustness) is reported beside it, not inside it.

| Method | Forecast right (daily surge) | Not needed (constant) | Sudden burst | Mean S (equal weights) | Worst % | Pattern shift (robustness) |
|---|---|---|---|---|---|---|
| Static LP (solved once) | 60.7 | 87.8 | 57.8 | 68.7 | 37.3 | 59.7 |
| Reactive LP coordinator | 52.5 | 97.4 | 50.4 | 66.8 | 22.2 | 52.5 |
| Battery Dijkstra (reactive) | 83.2 | 88.6 | 83.1 | 85.0 | 74.3 | 83.0 |
| MADII (not retrained) | 13.0 | 15.5 | 15.6 | 14.7 | 2.9 | 13.6 |
| Predictive Dijkstra, Holt-Winters | 81.4 | 88.6 | 82.3 | 84.1 | 63.4 | 82.6 |
| Lifetime Dijkstra, Holt-Winters (load) | 82.3 | 88.6 | 81.2 | 84.0 | 68.6 | 82.4 |
| Lifetime Dijkstra, Holt-Winters (tau) | 83.0 | 86.8 | 83.4 | 84.4 | 75.8 | 83.4 |
| Forecast LP, persistence | 13.4 | 97.4 | 12.7 | 41.2 | 0.8 | 13.4 |
| Forecast LP, seasonal-naive | 89.8 | 97.4 | 86.5 | 91.2 | 66.7 | 88.1 |
| Forecast LP, Holt-Winters (a priori) | 82.3 | 97.4 | 81.0 | 86.9 | 62.8 | 74.9 |
| Forecast LP, Holt-Winters tuned (ours) | 89.8 | 97.4 | 88.3 | 91.8 | 74.6 | 87.9 |
| Forecast LP, perfect forecast (diagnostic) | 89.0 | 97.4 | 89.2 | 91.8 | 76.3 | 89.0 |
| Forecast LP v2, Holt-Winters (ours) | 96.4 | 98.5 | 96.2 | 97.0 | 92.1 | 95.6 |
| Forecast LP v2, history mean (no forecast) | 95.7 | 98.5 | 95.2 | 96.5 | 84.8 | 95.1 |
| Forecast LP v2, perfect forecast (diagnostic) | 96.1 | 98.5 | 96.2 | 96.9 | 91.4 | 96.1 |
| Battery Dijkstra, receiver-weighted | 83.7 | 88.3 | 83.0 | 85.0 | 76.1 | 84.5 |
| Price-guided Dijkstra (daily LP prices) | 84.6 | 89.2 | 84.3 | 86.0 | 76.3 | 84.0 |
| Load-aware Dijkstra (ours, no LP) | 90.2 | 93.0 | 89.2 | 90.8 | 76.8 | 90.9 |
| Load table only (ablation, no battery) | 46.1 | 42.5 | 46.5 | 45.1 | 28.7 | 46.3 |
| Load-aware Dijkstra, LEST load table (4 tiers) | 89.5 | 92.6 | 89.3 | 90.5 | 76.8 | 90.0 |
| EBR-DA-style (energy + load cost, reconstructed) | 50.6 | 49.7 | 50.6 | 50.3 | 37.1 | 50.4 |

Paired vs Battery Dijkstra (reactive) (wins/ties/losses, Wilcoxon p):

- Static LP (solved once): R: 0/1/29, p=2.6e-06; N: 16/0/14, p=0.82; B: 0/0/30, p=1.9e-09; D: 0/0/30, p=1.9e-09
- Reactive LP coordinator: R: 0/0/30, p=1.9e-09; N: 30/0/0, p=1.9e-09; B: 1/1/28, p=3.5e-06; D: 1/0/29, p=3.7e-09
- MADII (not retrained): R: 0/0/30, p=1.9e-09; N: 0/0/30, p=1.9e-09; B: 0/0/30, p=1.9e-09; D: 0/0/30, p=1.9e-09
- Predictive Dijkstra, Holt-Winters: R: 8/5/17, p=0.046; N: 0/30/0, p=1; B: 7/10/13, p=0.12; D: 9/2/19, p=0.25
- Lifetime Dijkstra, Holt-Winters (load): R: 9/5/16, p=0.09; N: 0/30/0, p=1; B: 9/6/15, p=0.054; D: 9/7/14, p=0.19
- Lifetime Dijkstra, Holt-Winters (tau): R: 7/11/12, p=0.52; N: 6/4/20, p=0.00015; B: 14/7/9, p=0.78; D: 13/6/11, p=0.61
- Forecast LP, persistence: R: 0/0/30, p=1.9e-09; N: 30/0/0, p=1.9e-09; B: 0/0/30, p=1.9e-09; D: 0/0/30, p=1.9e-09
- Forecast LP, seasonal-naive: R: 28/1/1, p=3.2e-06; N: 30/0/0, p=1.9e-09; B: 22/0/8, p=0.0024; D: 22/2/6, p=0.00042
- Forecast LP, Holt-Winters (a priori): R: 16/1/13, p=0.87; N: 30/0/0, p=1.9e-09; B: 13/4/13, p=0.38; D: 7/1/22, p=0.00032
- Forecast LP, Holt-Winters tuned (ours): R: 28/1/1, p=4.8e-06; N: 30/0/0, p=1.9e-09; B: 24/1/5, p=0.00011; D: 23/2/5, p=0.00075
- Forecast LP, perfect forecast (diagnostic): R: 26/2/2, p=1.4e-05; N: 30/0/0, p=1.9e-09; B: 24/3/3, p=2.9e-05; D: 25/1/4, p=0.00019
- Forecast LP v2, Holt-Winters (ours): R: 30/0/0, p=1.9e-09; N: 30/0/0, p=1.9e-09; B: 30/0/0, p=1.7e-06; D: 30/0/0, p=1.9e-09
- Forecast LP v2, history mean (no forecast): R: 30/0/0, p=1.9e-09; N: 30/0/0, p=1.9e-09; B: 30/0/0, p=1.9e-09; D: 30/0/0, p=1.9e-09
- Forecast LP v2, perfect forecast (diagnostic): R: 30/0/0, p=1.7e-06; N: 30/0/0, p=1.9e-09; B: 30/0/0, p=1.9e-09; D: 30/0/0, p=1.9e-09
- Battery Dijkstra, receiver-weighted: R: 15/7/8, p=0.094; N: 10/4/16, p=0.73; B: 8/11/11, p=0.84; D: 16/6/8, p=0.063
- Price-guided Dijkstra (daily LP prices): R: 17/7/6, p=0.039; N: 17/4/9, p=0.041; B: 15/3/12, p=0.12; D: 17/8/5, p=0.067
- Load-aware Dijkstra (ours, no LP): R: 29/1/0, p=2.6e-06; N: 30/0/0, p=1.7e-06; B: 29/1/0, p=2.6e-06; D: 28/1/1, p=2.8e-06
- Load table only (ablation, no battery): R: 0/0/30, p=1.9e-09; N: 0/0/30, p=1.9e-09; B: 0/0/30, p=1.9e-09; D: 0/0/30, p=1.9e-09
- Load-aware Dijkstra, LEST load table (4 tiers): R: 29/1/0, p=2.6e-06; N: 30/0/0, p=1.9e-09; B: 30/0/0, p=1.7e-06; D: 29/0/1, p=5.6e-09
- EBR-DA-style (energy + load cost, reconstructed): R: 0/0/30, p=1.7e-06; N: 0/0/30, p=1.9e-09; B: 0/0/30, p=1.7e-06; D: 0/0/30, p=1.9e-09

Ablations (first vs second, wins/ties/losses, Wilcoxon p):

- Forecast LP v2, Holt-Winters (ours) vs Forecast LP, Holt-Winters tuned (ours) (what the planner fixes add): R: +6.6 pts, 30/0/0, p=1.9e-09; N: +1.1 pts, 24/2/4, p=0.00013; B: +7.8 pts, 30/0/0, p=1.7e-06; D: +7.7 pts, 30/0/0, p=1.9e-09
- Forecast LP v2, Holt-Winters (ours) vs Forecast LP v2, history mean (no forecast) (what the forecast adds): R: +0.7 pts, 17/6/7, p=0.047; N: +0.0 pts, 0/30/0, p=1; B: +1.0 pts, 14/6/10, p=0.13; D: +0.5 pts, 13/8/9, p=0.22
- Forecast LP v2, perfect forecast (diagnostic) vs Forecast LP v2, Holt-Winters (ours) (room left for a better forecast): R: -0.3 pts, 6/11/13, p=0.35; N: +0.0 pts, 0/30/0, p=1; B: -0.0 pts, 13/6/11, p=0.81; D: +0.5 pts, 15/9/6, p=0.2

Break-even forecast accuracy (G = gain when the forecast is right; C = loss when not needed / burst):

- Predictive Dijkstra, Holt-Winters vs Battery Dijkstra (reactive): G = -1.8, C(N) = +0.0, C(B) = +0.8 -> no gain when the forecast is right, so prediction does not pay off as a forecast
- Predictive Dijkstra, Holt-Winters vs Reactive LP coordinator: G = +28.9, C(N) = +8.8, C(B) = -31.9 -> p* = 0 on equal weights: it loses in one scenario, but its gains elsewhere outweigh that loss
- Lifetime Dijkstra, Holt-Winters (load) vs Battery Dijkstra (reactive): G = -0.9, C(N) = +0.0, C(B) = +1.9 -> no gain when the forecast is right, so prediction does not pay off as a forecast
- Lifetime Dijkstra, Holt-Winters (load) vs Reactive LP coordinator: G = +29.8, C(N) = +8.8, C(B) = -30.9 -> p* = 0 on equal weights: it loses in one scenario, but its gains elsewhere outweigh that loss
- Lifetime Dijkstra, Holt-Winters (tau) vs Battery Dijkstra (reactive): G = -0.2, C(N) = +1.8, C(B) = -0.3 -> no gain when the forecast is right, so prediction does not pay off as a forecast
- Lifetime Dijkstra, Holt-Winters (tau) vs Reactive LP coordinator: G = +30.5, C(N) = +10.6, C(B) = -33.0 -> p* = 0 on equal weights: it loses in one scenario, but its gains elsewhere outweigh that loss
- Forecast LP, persistence vs Battery Dijkstra (reactive): G = -69.9, C(N) = -8.8, C(B) = +70.4 -> no gain when the forecast is right, so prediction does not pay off as a forecast
- Forecast LP, persistence vs Reactive LP coordinator: G = -39.2, C(N) = +0.0, C(B) = +37.7 -> no gain when the forecast is right, so prediction does not pay off as a forecast
- Forecast LP, seasonal-naive vs Battery Dijkstra (reactive): G = +6.5, C(N) = -8.8, C(B) = -3.4 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP, seasonal-naive vs Reactive LP coordinator: G = +37.2, C(N) = +0.0, C(B) = -36.1 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP, Holt-Winters (a priori) vs Battery Dijkstra (reactive): G = -1.0, C(N) = -8.8, C(B) = +2.2 -> no gain when the forecast is right, so prediction does not pay off as a forecast
- Forecast LP, Holt-Winters (a priori) vs Reactive LP coordinator: G = +29.7, C(N) = +0.0, C(B) = -30.6 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP, Holt-Winters tuned (ours) vs Battery Dijkstra (reactive): G = +6.5, C(N) = -8.8, C(B) = -5.2 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP, Holt-Winters tuned (ours) vs Reactive LP coordinator: G = +37.2, C(N) = +0.0, C(B) = -38.0 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP, perfect forecast (diagnostic) vs Battery Dijkstra (reactive): G = +5.7, C(N) = -8.8, C(B) = -6.1 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP, perfect forecast (diagnostic) vs Reactive LP coordinator: G = +36.5, C(N) = +0.0, C(B) = -38.8 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP v2, Holt-Winters (ours) vs Battery Dijkstra (reactive): G = +13.2, C(N) = -9.9, C(B) = -13.1 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP v2, Holt-Winters (ours) vs Reactive LP coordinator: G = +43.9, C(N) = -1.1, C(B) = -45.8 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP v2, history mean (no forecast) vs Battery Dijkstra (reactive): G = +12.4, C(N) = -9.9, C(B) = -12.1 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP v2, history mean (no forecast) vs Reactive LP coordinator: G = +43.2, C(N) = -1.1, C(B) = -44.8 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP v2, perfect forecast (diagnostic) vs Battery Dijkstra (reactive): G = +12.9, C(N) = -9.9, C(B) = -13.0 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP v2, perfect forecast (diagnostic) vs Reactive LP coordinator: G = +43.6, C(N) = -1.1, C(B) = -45.8 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Price-guided Dijkstra (daily LP prices) vs Battery Dijkstra (reactive): G = +1.3, C(N) = -0.6, C(B) = -1.1 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Price-guided Dijkstra (daily LP prices) vs Reactive LP coordinator: G = +32.1, C(N) = +8.2, C(B) = -33.9 -> p* = 0 on equal weights: it loses in one scenario, but its gains elsewhere outweigh that loss
