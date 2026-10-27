"""Evaluate corrected-reward MADII with per-deployment inference seeds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

import env as E
import policies as P
from evaluate import load
from lp_bound import max_lifetime_T
from train import greedy_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--output', default='runs/correct_reward_seed0/evaluation.json')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seeds = list(range(args.seeds))
    ceilings = np.array([max_lifetime_T(E.WSN(seed=s)) for s in seeds])
    baseline = [E.run_episode(E.WSN(seed=s), P.battery_weighted) for s in seeds]
    ref = np.array([r[0.01]['SR'] for r in baseline])
    result = {'test_seeds': seeds, 'inference_seed_rule': '1000000 + 100000*repeat + deployment_seed',
              'reward': 'current round start energy minus end energy',
              'ceiling_rounds': ceilings.tolist(),
              'battery_dijkstra': {'FND': ref.tolist(), 'mean_FND': float(ref.mean()),
                                   'mean_pct_ceiling': float(((ref-1)/ceilings*100).mean())},
              'models': {}}
    for path in args.checkpoints:
        net, cfg = load(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        repeats = []
        for repeat in range(args.repeats):
            rows = []
            for seed in seeds:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(1000000 + 100000*repeat + seed)
                    row = E.run_episode(E.WSN(seed=seed), lambda w: greedy_action(net, w))
                rows.append({str(b): row[b] for b in row})
            repeats.append(rows)
            print(json.dumps({'checkpoint':path, 'repeat':repeat,
                              'mean_FND':float(np.mean([r['0.01']['SR'] for r in rows]))}), flush=True)
        fnd = np.array([[r['0.01']['SR'] for r in rows] for rows in repeats], float)
        hnd = np.array([[r['0.5']['SR'] for r in rows] for rows in repeats], float)
        pct = (fnd-1)/ceilings[None,:]*100
        paired = fnd.mean(axis=0)-ref
        result['models'][Path(path).stem] = {
            'checkpoint':path, 'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'episode':checkpoint['ep'], 'training_args':cfg, 'raw':repeats,
            'mean_FND':float(fnd.mean()), 'mean_HND':float(hnd.mean()),
            'mean_pct_ceiling':float(pct.mean()), 'primary_repeat_mean_FND':float(fnd[0].mean()),
            'mean_FND_per_repeat':fnd.mean(axis=1).tolist(),
            'W_T_L_vs_battery':[(paired>0).sum().item(),(paired==0).sum().item(),(paired<0).sum().item()],
            'wilcoxon_p_vs_battery':float(wilcoxon(paired).pvalue) if np.any(paired) else 1.0}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# MADII corrected-reward rerun', '',
             '30 held-out deployment seeds; three fixed inference repeats per model.', '',
             '| Model | Training episode | Mean FND | Mean % of ceiling | Mean HND |',
             '|---|---:|---:|---:|---:|',
             f"| Battery Dijkstra | — | {ref.mean():.2f} | {result['battery_dijkstra']['mean_pct_ceiling']:.2f} | {np.mean([r[0.5]['SR'] for r in baseline]):.2f} |"]
    for name,r in result['models'].items():
        lines.append(f"| {name} | {r['episode']} | {r['mean_FND']:.2f} | {r['mean_pct_ceiling']:.2f} | {r['mean_HND']:.2f} |")
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
