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
- Implement EBR-DA's link cost as an extra baseline. It is the closest competitor, and beating
  it on the same bench would make the method claim much stronger.
- Search IEEE Xplore and Scopus directly (not reachable from this environment) for
  "load-balanced tree" + "maximum lifetime" + "sequential" or "greedy".
