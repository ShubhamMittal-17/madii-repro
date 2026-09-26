import numpy as np, env as E, dijkstra_rl as D
from multiprocessing import Pool
VAL = range(500, 520)
def run(args):
    kind, k, s = args
    w = E.WSN(seed=s)
    for _ in range(1000):
        base = D.battery_mult(w, k if kind == "alpha" else 1.0)
        if kind == "lifetime":
            phi = D.node_features(w, D.dijkstra_tree(w, base))
            base = base * np.exp(k * (phi[:, 5]))          # penalise bottleneck nodes
        if w.step(D.dijkstra_tree(w, base))["n_dead"] >= 1: break
    return w.round / D.ceiling(s)
if __name__ == "__main__":
    with Pool(4) as p:
        for kind, ks in [("alpha", [0.5, 1, 1.5, 2, 3, 4]), ("lifetime", [0.5, 1, 2, 4])]:
            for k in ks:
                r = p.map(run, [(kind, k, s) for s in VAL]); print(kind, k, round(np.mean(r), 4), round(min(r), 3), flush=True)
