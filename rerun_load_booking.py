"""Rerun affected routing methods with fixed parameters and original LP bounds."""
import argparse
import copy
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import env as E
import intel as I
import traffic as TR
import dijkstra_rl as D
from predictive import LESTDijkstra, LoadAwareDijkstra

OUT = Path('runs/correct_load_booking')
LEST = 'Load-aware Dijkstra, LEST load table (4 tiers)'
EXACT = 'Load-aware Dijkstra (ours, no LP)'
LOAD = 'Load table only (ablation, no battery)'
BATT = 'Battery Dijkstra (reactive)'


def build(name, w, h):
    if name == LEST:
        return LESTDijkstra(w,h,kappa=1.,order='near',levels='lest',rule='original',band=.05,exact_energy=True)
    if name == EXACT:
        return LoadAwareDijkstra(w,h,kappa=1.,order='near')
    if name == LOAD:
        return LoadAwareDijkstra(w,h,kappa=1.,order='near',energy=False)
    return D.fast_battery_weighted


def episode(w, pol, length, oracle):
    failed = False
    while w.round < length:
        if w.step(pol(w))['n_dead'] >= 1:
            failed = True
            break
    return {'failure_round':w.round if failed else None,
            'observed_rounds':w.round, 'censored':not failed,
            'legacy_scored_rounds':w.round-1,
            'pct_oracle':100*(w.round-1)/oracle,
            'control_bits_per_round':w.LC, 'data_packet_bits':w.L}


def synthetic_job(args):
    sc, seed, oracle = args
    w0 = E.WSN(seed=seed)
    h,tr = TR.generate(sc,w0.pos[:w0.n],seed+1000)
    out = {'oracle':oracle, 'methods':{}}
    for name in [BATT,LEST,EXACT,LOAD]:
        w = E.WSN(seed=seed,traffic=tr)
        out['methods'][name] = episode(w,build(name,w,h),len(tr),oracle)
    return sc,seed,out


def intel_job(args):
    e0,start,oracle = args
    tr = np.load('data/intel_lab/traffic_dT0.5_dL100.npy')
    h,fut = tr[:start],tr[start:]
    out = {'oracle':oracle,'methods':{}}
    for name in ['Battery Dijkstra',LEST,EXACT]:
        w = I.IntelWSN(e0=e0,traffic=fut)
        out['methods'][name] = episode(w,build(name,w,h),len(fut),oracle)
    return e0,start,out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['synthetic','intel'],required=True)
    ap.add_argument('--workers',type=int,default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.phase == 'synthetic':
        original = json.loads(Path('results_dynamic.json').read_text())
        jobs = [(sc,s,original[sc][str(s)]['oracle']) for sc in 'RNBD' for s in range(30)]
        raw,merged = {},copy.deepcopy(original)
        with (OUT/'synthetic_progress.jsonl').open('w') as log,Pool(a.workers) as pool:
            for done,(sc,s,r) in enumerate(pool.imap_unordered(synthetic_job,jobs),1):
                raw.setdefault(sc,{})[str(s)] = r
                for m,v in r['methods'].items():
                    if m == BATT:
                        assert abs(v['pct_oracle']-original[sc][str(s)][m])<1e-8, ('baseline changed',sc,s)
                    merged[sc][str(s)][m] = v['pct_oracle']
                log.write(json.dumps({'scenario':sc,'seed':s,'result':r})+'\n');log.flush()
                if done%10==0:
                    print(json.dumps({'phase':a.phase,'completed':done,'total':len(jobs)}),flush=True)
        (OUT/'synthetic_raw.json').write_text(json.dumps(raw,indent=2)+'\n')
        (OUT/'results_dynamic_corrected.json').write_text(json.dumps(merged,indent=2)+'\n')
    else:
        originals={1.:json.loads(Path('results_intel_e1.json').read_text()),
                   1.5:json.loads(Path('results_intel.json').read_text())}
        starts=list(range(336,545,8))
        jobs=[(e0,s,d['oracle'][str(s)]) for e0,d in originals.items() for s in starts]
        raw={}
        with (OUT/'intel_progress.jsonl').open('w') as log,Pool(a.workers) as pool:
            for done,(e0,s,r) in enumerate(pool.imap_unordered(intel_job,jobs),1):
                raw.setdefault(str(e0),{})[str(s)]=r
                old=originals[e0]['L']['Battery Dijkstra'][starts.index(s)]
                assert abs(r['methods']['Battery Dijkstra']['pct_oracle']-old)<1e-8, ('baseline changed',e0,s)
                log.write(json.dumps({'energy':e0,'start':s,'result':r})+'\n');log.flush()
                if done%5==0:
                    print(json.dumps({'phase':a.phase,'completed':done,'total':len(jobs)}),flush=True)
        (OUT/'intel_raw.json').write_text(json.dumps(raw,indent=2)+'\n')
        for e0,d in originals.items():
            merged=copy.deepcopy(d)
            for m in ['Battery Dijkstra',LEST,EXACT]:
                merged['L'][m]=[raw[str(e0)][str(s)]['methods'][m]['pct_oracle'] for s in starts]
                merged['censored'][m]=sum(raw[str(e0)][str(s)]['methods'][m]['censored'] for s in starts)
            (OUT/f'results_intel_e{e0:g}_corrected.json').write_text(json.dumps(merged,indent=2)+'\n')
    print('Completed',a.phase,flush=True)


if __name__ == '__main__':
    main()
