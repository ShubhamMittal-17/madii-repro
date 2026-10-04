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

## Update: full texts read (docs/papers/)

- **MADII**: J. Yang, W. Li, C. Li, L. Zhang, L. Liu, "An Energy-Efficient and Transmission-
  Efficient Adaptive Routing Algorithm Using Deep Reinforcement Learning for Wireless Sensor
  Networks", IEEE Internet of Things Journal 12(23):50414-50426, Dec. 2025,
  doi:10.1109/JIOT.2025.3609624.
  - Corresponding author: Cuiran Li, licr@mail.lzjtu.cn (Lanzhou Jiaotong University).
  - **No code link** in the paper.
  - **Baselines (Sec. IV-A):** FCM, GWO-WOA, HBA, POA, SCSO, MADTI, MADIE. These are clustering,
    metaheuristic and learned methods. There is **no shortest-path / Dijkstra baseline and no
    optimum bound**, confirmed from the full text.
  - Setup: 100 nodes, 500 x 500 m, no data compression, FND and HND, one RTX 4070 Ti GPU.
- **Shan, Liang, Luo, Shen**, "Network lifetime maximization for time-sensitive data gathering in
  wireless sensor networks", Computer Networks 57 (2013) 1063-1077,
  doi:10.1016/j.comnet.2012.12.005.
  - A load-balanced **shortest-path** spanning tree: every sensor reaches the sink in its minimum
    hop count.
  - Non-aggregated relaying; a node's energy is proportional to its number of descendants.
  - Top-down network-flow construction plus distributed balance refinement.
  - Lifetime is at least **85% of the upper bound**.
  - Differences from ours: one tree under a min-hop constraint, against our tree rebuilt every
    round from batteries and the load booked that round.
- **Chang & Tassiulas**, "Maximum lifetime routing in wireless sensor networks", IEEE/ACM
  Transactions on Networking 12(4), 2004 (pdf in docs/papers). The source of our oracle LP.

## Recent work (2023-2025) to cite alongside the classics

- Learned routing keeps being proposed and compared with weak baselines:
  - multi-agent DRL WSN routing, US patent 12,381,811;
  - DRL-OLSR / SOM-OLSR for mobile WSNs (IET Networks, 2024);
  - energy-efficient DQN routing for wireless IoT (IJEECS, 2024).
- "Learning for routing: A guided review of recent developments and future directions",
  arXiv:2507.00218 (2025). A recent survey to anchor the related-work section.
- "When Simple Model Just Works: Is Network Traffic Classification in Crisis?", arXiv:2506.08655
  (2025). In traffic classification, a 1-NN baseline matches or beats state-of-the-art deep
  models. The same pattern as our finding, in a neighbouring networking problem.

## Update: two more papers found by the authors (docs/papers/)

- **M. U. Younus et al., "Optimizing the Lifetime of Software Defined Wireless Sensor Network via
  Reinforcement Learning", IEEE Access 9, 2021 (published Dec. 2020), doi:10.1109/ACCESS.2020.3046693.**
  - RL runs in an SDN controller that has the *global* network view and computes the routing
    tables, on a **real testbed**.
  - Four reward functions; loop-free candidate paths come from spanning-tree protocol.
  - **Compared only with an RL-based WSN routing scheme**: +23-30% lifetime. Dijkstra appears
    only in related work (a cited Q-learning paper beat Dijkstra on congestion), not as a baseline.
  - No optimum bound.
  - **Relevance:** the same architecture as our sink-computed routers (a central controller with
    global state that pushes routes). It shows the deployment model is realistic, and it is
    another learned method with no strong classical baseline.
- **"Energy Efficient Data Transmission in WSN for Network Lifetime Enhancement", Proc. 7th
  ICMCSI-2026, IEEE (pp. 183-189).**
  - A comparison paper. It tabulates published results of about ten methods (DBN routing,
    ELPSO-PSO-BPNN, TIOCHR, SAE-PNN, model-free DRL, ...) on two datasets: a 10-node
    network-feature dataset and a patient health-monitoring dataset.
  - Values are in mixed units (energy from 0.35 J to 99 J; lifetime as a %). There is no common
    simulator, no shortest-path baseline and no optimum.
  - **Relevance:** current (2026) evidence that WSN lifetime methods are compared without a
    common benchmark or bound. It motivates our oracle-normalised bench; not a method competitor.
