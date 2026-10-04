# Literature check: is load-aware / LEST Dijkstra new?

Search done October 2026 (web search only; the full texts of the papers below could not be
opened from this environment, so details come from abstracts and should be confirmed by
reading each paper before the report cites them).

## Closest prior work

| Work | What it does | Relation to ours |
|---|---|---|
| O. A. Mahdi et al., "An Energy-Aware and Load-balancing Routing scheme for Wireless Sensor Networks" (EBR-DA), IJEECS 12(3), 1312-1319, 2018, doi:10.11591/ijeecs.v12.i3.pp1312-1319 | Modified Dijkstra over a link cost whose node weight combines residual energy and load; routing tree rebuilt as events occur. Compared with InFRA and DRINA. | **Same idea family**: energy + load in a Dijkstra cost is not new. No optimum bound reported in the abstract. |
| W. Liang et al., load-balanced shortest routing tree for time-sensitive data gathering (top-down layer-by-layer construction via network flow, then distributed balance refinement) | Builds a load-balanced shortest-path tree; NP-hard problem, heuristic. | Reports **~85% of the optimum lifetime**, so it is measured against the optimum, as we are. Different setting (shortest-hop constraint), so the numbers are not directly comparable. |
| DECOR: distributed construction of load-balanced routing trees for many-to-one sensor networks | Distributed load-balanced tree construction. | Prior art for load-balanced trees built without a central planner. |
| J.-H. Chang and L. Tassiulas, "Maximum lifetime routing in wireless sensor networks", IEEE/ACM Trans. Networking 12(4), 2004 | The max-lifetime LP (our oracle) plus flow augmentation: shortest paths with link costs from energy use and residual energy at both ends; reported near-optimal. | Our oracle and planner come from here. Our sweep found stronger residual-energy exponents hurt **when each round routes on a single tree**. That differs from their incremental flow augmentation, and is worth stating. |
| Hysteresis-driven routing for energy-harvesting networks; quantised residual-energy "grades" for relay selection (several papers) | Route switching with hysteresis thresholds; residual energy shared as discrete levels. | Tiered state and hysteresis in routing are **not new** in themselves. |
| Multi-agent DRL routing for WSNs (e.g. US patent 12,381,811; DQN routing papers) | Claim longer lifetime than shortest-path routing, LEACH or fuzzy C-means. | Supports our critique: the learned methods are compared with weak baselines, not with battery-weighted Dijkstra or an optimum bound. |

## What we can still claim

1. **The benchmark finding (strongest, appears new):** measured against the provable LP optimum,
   the four methods line up as follows (synthetic and real Intel Lab traffic):
   - a published learned router (MADII): ~15%
   - battery-weighted Dijkstra: ~85%
   - LEST load-table Dijkstra: ~90.5%
   - a per-round LP planner: ~97%

   We found no paper ranking learned WSN routing against these classical baselines and the
   optimum.
2. **Within-round load booking:** the tree is built node by node, nearest the sink first, and
   each node is charged for the load already routed through a relay *in the same round*. The
   EBR-DA abstract describes load in the node weight, but not this sequential construction.
   Confirm against the full text before claiming it.
3. **LEST-style sharing of the load table:** 4 tiers, 1-byte piggybacked reports and a heartbeat
   snapshot, with the overhead charged in the energy model. It costs 0.3 points against exact
   loads.
4. **Ablations and diagnostics:**
   - both tables are needed: load only 45.1%, energy only 85.0%, both 90.5%;
   - energy must stay exact (in 4 tiers it falls to 79%);
   - LEST's original trigger freezes slowly draining quantities but works for load.
5. **Negative results:** forecasting inside Dijkstra, lifetime weighting, exponent tuning and LP
   prices each stay within about ±1 point of battery Dijkstra. Prediction adds under 1 point
   even to the LP planner.

## What we should not claim

- That combining energy and load in a Dijkstra cost is new (EBR-DA, 2018).
- That load-balanced routing trees are new (Liang et al.; DECOR).
- That tiered or hysteresis state sharing is new in itself.

## To do before submission

- Read EBR-DA, Liang et al. and DECOR in full and confirm the differences above.
- ~~Implement EBR-DA's link cost as an extra baseline.~~ Done, as a reconstruction from the
  published description (predictive.EBRDA, tuned on validation, tune_ebrda.py). Held-out mean
  50.3% of the optimum, against 85.0% for battery Dijkstra and 90.5% for LEST load-table
  Dijkstra (loses all 120 deployments to battery Dijkstra). Its linear energy term
  (1 - E/E0) barely reacts as a node nears empty, whereas E0/E rises sharply. Re-check against
  the full paper: if its cost differs, rerun.
- Search IEEE Xplore and Scopus directly (not reachable from this environment) for
  "load-balanced tree" + "maximum lifetime" + "sequential" or "greedy".


## Update: EBR-DA implemented from the full paper

docs/papers/Mahdi2018_EBR-DA.pdf, Sec. 3-4. The paper specifies:
- node weight NW = 0.6 (1 - Eres/Einit)^2 + 0.4 (1 - Bava/Btotal)^2;
- a hop tree over an 80 m communication radius in a 500 m x 500 m field;
- each node forwards to the lightest-weight neighbour one hop nearer the sink.

The paper evaluates EBR-DA in MATLAB against DRINA and InFRA only (energy consumption, no
optimum bound), and its design assumes in-network data aggregation, which our energy model
(like MADII's) does not have.

Two approximations in our implementation (`predictive.EBRDAPaper`):
- **Buffer occupancy:** our model has no buffers, so occupancy is taken as the node's
  last-round traffic over the busiest node's.
- **Bridging:** a node with no neighbour in range forwards to its nearest node nearer the sink.

Held-out results (30 deployments x 4 scenarios, % of the oracle):

| Method | Mean | Worst |
|---|---|---|
| EBR-DA as specified | 13.9 | 0.5 |
| EBR-DA, isolated nodes bridged | 15.1 | 0.5 |
| EBR-DA-style reconstruction (tuned Dijkstra cost) | 50.3 | 37.1 |
| Battery Dijkstra | 85.0 | 74.3 |
| LEST load-table Dijkstra | 90.5 | 76.8 |

Why EBR-DA scores so low here:
- **Hop-minimal routing** sends all traffic through the ring of about 12 nodes within 80 m of
  the sink.
- **Its squared weights start near 0**, so early choices are effectively shortest-hop.

Without aggregation, that ring drains quickly. State this caveat when citing: EBR-DA is shown
outside the aggregation setting it was designed for.
