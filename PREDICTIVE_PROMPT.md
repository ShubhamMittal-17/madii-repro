# Build prompt: Predictive battery-Dijkstra with Holt-Winters forecasting

Hand this whole file to the implementer (person or agent). It is self-contained.

---

## Context (read first)

This repo (`madii-repro`) reproduces MADII (Yang et al., IEEE IoT-J 12(23), Dec 2025),
a deep-RL WSN router, and compares it with classical routing against a provable
max-lifetime LP ceiling. Established results, static traffic (every node sends one
4 kbit packet per round), 30 held-out deployments:

| method | % of LP ceiling (first node death) |
|---|---|
| LP re-solved every 1-4 rounds (classical) | ~97-98% (5 deployments) |
| adaptive LP-flow, re-solve every 8 (classical) | 94.5% |
| battery-weighted Dijkstra (classical) | 88.4% |
| RL warm-started from Dijkstra (`dijkstra_rl.py`) | 88.9% (not significant, p=0.31) |
| MADII reproduction v3 (`checkpoints/madii_v3_best.pt`) | 11.1% |

Conclusion so far: with static traffic, learning is unnecessary. The open question is
**time-varying traffic**, where a router that only sees the present can spend a relay's
battery just before that relay is needed.

**The idea to build:** keep battery-weighted Dijkstra as the router, and add a
Holt-Winters forecast of each node's future traffic. Dijkstra then routes on each node's
*effective* energy, which is its battery minus a reserve for its own forecast load.
Relays that will be needed later are spared now. Forecasting is the only learned or
fitted part; routing stays classical.

## Existing code to reuse (do not rewrite)

- `env.py` - `WSN` round simulator (radio model Eq 1-2, round rules R1-R4, `run_episode`).
- `dijkstra_rl.py` - `dijkstra_tree(w, node_mult)` (fast scipy Dijkstra; cost of i->j is
  `base(i,j) * node_mult[i]`), `battery_mult`, `fast_battery_weighted`, `ceiling(seed)`.
- `lp_bound.py` - `max_lifetime_T(env, alive, energy, return_flow)`.
- `lp_flow.py` - `LPFlowRouting`, `LPFlowAdaptive` (reactive LP baselines).
- `compare.py` - paired, 30-deployment evaluation harness (extend, don't fork).
- `test_env.py` - add tests here.

## Step 1 - Dynamic traffic in the simulator

Extend `WSN` so each alive node i sends `k_i(t)` packets in round t (an integer >= 0)
instead of exactly one. Traffic comes from a `traffic[t, i]` array passed to the
constructor. `traffic=None` must reproduce today's behaviour exactly, so every existing
number and test stays unchanged (add a regression test for this).

- A node's load is its own `k_i(t) * L` bits plus everything relayed through it (R1-R4
  unchanged).
- Delivery accounting (b_succ/b_fail/b_retry) is per packet.

Traffic generators (`traffic.py`), each returning `traffic[T, n]`:

- **R - regular pattern:** a per-node daily cycle, e.g.
  `k_i(t) ~ Poisson(base_i * (1 + A_i * sin(2*pi*(t - phase_i)/P)))`, where P is the
  number of rounds per "day" (e.g. P = 24). Hotspot nodes (a spatial cluster) get a large A_i.
- **N - no pattern:** a constant rate, `k_i(t) ~ Poisson(base_i)`.
- **B - sudden bursts:** the R pattern plus unforecastable bursts. With probability
  p_burst per round, pick a random node (including nodes that are normally only relays),
  multiply its rate by M (e.g. 5x) for D rounds.
- Every generator takes a seed and is deterministic given it.
- Later (Step 7): `traffic_intel.py` builds `traffic[T, n]` from the Intel Berkeley Lab
  data by event-driven reporting.

## Step 2 - Holt-Winters forecaster (`forecast.py`)

Additive Holt-Winters with season length P, one model per node, updated online each round
with that node's observed `k_i(t)`:

```
level_t  = a * (y_t - season_{t-P})           + (1 - a) * (level_{t-1} + trend_{t-1})
trend_t  = b * (level_t - level_{t-1})        + (1 - b) * trend_{t-1}
season_t = g * (y_t - level_t)                + (1 - g) * season_{t-P}
forecast y_{t+h} = level_t + h * trend_t + season_{t+h-P*ceil(h/P)}
```

- Vectorise across nodes (numpy arrays of shape (n,)); never loop over nodes in Python.
- Warm-up: the first 2P rounds use the running mean; initialise the season from the
  first P rounds.
- Uncertainty: keep a rolling window of absolute 1-step errors per node, and output an
  upper quantile `q_hi(i, t+h) = forecast + z * MAE_i * sqrt(h)`; z sets the quantile.
- Fit (a, b, g) once, by grid search on a separate training traffic seed. Never fit on
  evaluation seeds.
- Interface: `hw.update(y_t)`, `hw.forecast(H) -> (mean[H, n], upper[H, n])`.
- Baselines behind the same interface: `Persistence` (next = now) and `SeasonalNaive`
  (next = same time one season ago).
- **No leakage:** at round t the forecaster may only have seen rounds < t.

## Step 3 - Predictive battery-Dijkstra (`predictive.py`)

A policy callable `PredictiveDijkstra(w, forecaster, H, lam, alpha=1.0)`, called once
per round:

1. Build the ordinary battery-weighted tree `p0 = fast_battery_weighted(w)`.
2. Get the forecast `upper[h, i]` for the next H rounds.
3. Estimate each node's future spend if routing stays like p0: the forecast packets of
   i's subtree times i's per-packet TX + RX cost on p0. Sum over the horizon to get
   `reserve_i`.
4. Effective energy: `E_eff_i = max(E_i - lam * reserve_i, eps * E0)`.
5. Route on the effective energy: `dijkstra_tree(w, (E0 / E_eff) ** alpha)`.

`lam` is the **trust dial**:
- `lam = 0` must reproduce reactive battery-Dijkstra exactly (add a test).
- Larger `lam` means stronger protection of nodes with a heavy future load.

Sweep `lam` in {0, 0.25, 0.5, 1, 2} and `H` in {P/4, P/2, P}.

(Optional, same interface: `PredictiveLP`, the multi-period version of `LPFlowRouting`
fed the forecast traffic.)

## Step 4 - Oracle ceiling for time-varying traffic (`lp_oracle.py`)

The LP with full knowledge of `traffic[t, i]`. Lifetime with time-varying traffic is not
linear in T, so bisect on T:

- `feasible(T)`: an LP with flow variables per period (group rounds into periods of
  length P/4 to keep it small). Each node's traffic in each period must reach the sink,
  and each node's total TX + RX + control energy summed over all periods must be <= E0.
- Largest feasible T = `T_oracle`. It is an upper bound for every method in that
  scenario.
- Validate: with constant traffic, `T_oracle` must match `max_lifetime_T`.

## Step 5 - Evaluation (`compare_dynamic.py`)

Rows (methods):
1. Static routing: the LP solved once (`LPFlowRouting`).
2. Reactive LP coordinator (`LPFlowAdaptive`, re-solve every 4).
3. Reactive battery-Dijkstra (`fast_battery_weighted`).
4. MADII v3 **zero-shot** (`checkpoints/madii_v3_best.pt`, no retraining; label it so).
5. **Predictive Dijkstra + Holt-Winters (ours)**, best (lam, H) chosen on validation
   seeds only.
6. Predictive Dijkstra with the Persistence / SeasonalNaive forecasters (ablations).

Columns (scenarios): R, N, B.

- Evaluation: 30 held-out deployment seeds x 1 traffic seed each.
- Validation (for lam, H, HW parameters): a disjoint set of seeds.

Metric per (method m, scenario s):
`L(m, s) = FND rounds of m / T_oracle` (mean, worst case, paired Wilcoxon vs row 3).

Combined score and break-even:
- `S(m) = p_R*L(m,R) + p_N*L(m,N) + p_B*L(m,B)`, with `p_R + p_N + p_B = 1`.
- `G = L(pred,R) - L(react,R)`, `C_N = L(react,N) - L(pred,N)`, `C_B = L(react,B) - L(pred,B)`.
- `p* = Cbar / (G + Cbar)`, where `Cbar = (p_N*C_N + p_B*C_B) / (p_N + p_B)`.
- **Report the full 3 x 6 table first.** Plot S(m) against p_R for p_R in [0, 1] (split
  the remainder between N and B by a stated ratio), and mark p*.
- Do **not** choose p values by hand for the headline number (Step 6 estimates them).

## Step 6 - Estimating p from traffic history (backtest)

On a traffic history (synthetic now, Intel later), run the forecaster online and classify
each forecast window:

- **R** if the realised traffic lies within the forecast band.
- **N** if the series shows no significant seasonal pattern (the seasonal amplitude is
  below a threshold relative to the noise).
- **B** if the realised traffic exceeds the upper band by more than a factor.

The fractions give `p_R_hat, p_N_hat, p_B_hat`. Report S at those values, and the
decision rule "turn prediction on iff p_R_hat > p*". Then verify the decision was correct
on held-out history.

## Step 7 - Real data: Intel Berkeley Lab

- Source: `http://db.csail.mit.edu/labdata/data.txt.gz` and `mote_locs.txt`. The host
  must be allowed in the environment's network settings, or the files uploaded.
- Traffic: event-driven reporting. Node i sends a packet in round t when its light (or
  temperature) reading changed by more than a threshold since its last report. Use 2-3
  thresholds and report all of them.
- Layout: the real mote positions, scaled up to 500 x 500 m so the radio regime matches
  MADII's (the lab is about 40 x 30 m, below d0 = 87.7 m, where routing barely matters).
  State this scaling.
- Missing readings are gaps in the data, not zero traffic; interpolate or mask them.

## Acceptance criteria (must all hold)

- All existing tests pass, and `traffic=None` reproduces the old results bit-for-bit.
- New tests: `lam = 0` equals reactive Dijkstra; `T_oracle` matches the static LP under
  constant traffic; no method's FND exceeds `T_oracle`; the forecaster sees no future
  data.
- `compare_dynamic.py` writes `results_dynamic.md` and `results_dynamic.json` with the
  table, S-vs-p_R data, p*, and p_hat from the backtest.
- Runtime on 4 CPU cores: under 1 hour for the synthetic scenarios.

## Report honestly

- If Predictive Dijkstra does **not** beat reactive in R, say so. That is a valid result
  ("prediction not needed").
- The MADII row is zero-shot on a different layout, so label it as such.
- The fair baseline is the *reactive* version of the same router, not MADII.
