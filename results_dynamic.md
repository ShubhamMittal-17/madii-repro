30 held-out deployments per scenario; L = lifetime to first node death, % of the oracle LP.

| Method | Forecast right (daily surge) | Not needed (constant) | Sudden burst | Mean S (equal weights) | Worst % |
|---|---|---|---|---|---|
| Static LP (solved once) | 60.7 | 87.8 | 57.8 | 68.7 | 37.3 |
| Reactive LP coordinator | 52.5 | 97.4 | 50.4 | 66.8 | 22.2 |
| Battery Dijkstra (reactive) | 83.2 | 88.6 | 83.1 | 85.0 | 74.3 |
| MADII (not retrained) | 13.0 | 15.5 | 15.6 | 14.7 | 2.9 |
| Predictive Dijkstra, Holt-Winters | 81.4 | 88.6 | 82.3 | 84.1 | 63.4 |
| Forecast LP, persistence | 13.4 | 97.4 | 12.7 | 41.2 | 0.8 |
| Forecast LP, seasonal-naive | 89.8 | 97.4 | 86.5 | 91.2 | 66.7 |
| Forecast LP, Holt-Winters (ours) | 82.3 | 97.4 | 81.0 | 86.9 | 62.8 |
| Forecast LP, perfect forecast (diagnostic) | 89.0 | 97.4 | 89.2 | 91.8 | 76.3 |

Paired vs Battery Dijkstra (reactive) (wins/ties/losses, Wilcoxon p):

- Static LP (solved once): R: 0/1/29, p=2.6e-06; N: 16/0/14, p=0.82; B: 0/0/30, p=1.9e-09
- Reactive LP coordinator: R: 0/0/30, p=1.9e-09; N: 30/0/0, p=1.9e-09; B: 1/1/28, p=3.5e-06
- MADII (not retrained): R: 0/0/30, p=1.9e-09; N: 0/0/30, p=1.9e-09; B: 0/0/30, p=1.9e-09
- Predictive Dijkstra, Holt-Winters: R: 8/5/17, p=0.046; N: 0/30/0, p=1; B: 7/10/13, p=0.12
- Forecast LP, persistence: R: 0/0/30, p=1.9e-09; N: 30/0/0, p=1.9e-09; B: 0/0/30, p=1.9e-09
- Forecast LP, seasonal-naive: R: 28/1/1, p=3.2e-06; N: 30/0/0, p=1.9e-09; B: 22/0/8, p=0.0024
- Forecast LP, Holt-Winters (ours): R: 16/1/13, p=0.87; N: 30/0/0, p=1.9e-09; B: 13/4/13, p=0.38
- Forecast LP, perfect forecast (diagnostic): R: 26/2/2, p=1.4e-05; N: 30/0/0, p=1.9e-09; B: 24/3/3, p=2.9e-05

Break-even forecast accuracy (G = gain when the forecast is right; C = loss when not needed / burst):

- Predictive Dijkstra, Holt-Winters vs Battery Dijkstra (reactive): G = -1.8, C(N) = +0.0, C(B) = +0.8 -> no gain when the forecast is right, so prediction does not pay off as a forecast
- Predictive Dijkstra, Holt-Winters vs Reactive LP coordinator: G = +28.9, C(N) = +8.8, C(B) = -31.9 -> p* = 0 on equal weights: it loses in one scenario, but its gains elsewhere outweigh that loss
- Forecast LP, persistence vs Battery Dijkstra (reactive): G = -69.9, C(N) = -8.8, C(B) = +70.4 -> no gain when the forecast is right, so prediction does not pay off as a forecast
- Forecast LP, persistence vs Reactive LP coordinator: G = -39.2, C(N) = +0.0, C(B) = +37.7 -> no gain when the forecast is right, so prediction does not pay off as a forecast
- Forecast LP, seasonal-naive vs Battery Dijkstra (reactive): G = +6.5, C(N) = -8.8, C(B) = -3.4 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP, seasonal-naive vs Reactive LP coordinator: G = +37.2, C(N) = +0.0, C(B) = -36.1 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP, Holt-Winters (ours) vs Battery Dijkstra (reactive): G = -1.0, C(N) = -8.8, C(B) = +2.2 -> no gain when the forecast is right, so prediction does not pay off as a forecast
- Forecast LP, Holt-Winters (ours) vs Reactive LP coordinator: G = +29.7, C(N) = +0.0, C(B) = -30.6 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP, perfect forecast (diagnostic) vs Battery Dijkstra (reactive): G = +5.7, C(N) = -8.8, C(B) = -6.1 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
- Forecast LP, perfect forecast (diagnostic) vs Reactive LP coordinator: G = +36.5, C(N) = +0.0, C(B) = -38.8 -> p* = 0: it wins in every scenario, so it pays at any forecast accuracy
