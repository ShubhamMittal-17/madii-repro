"""Summarize corrected routing results without treating scenarios as independent."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from rerun_load_booking import OUT,LEST,EXACT,LOAD,BATT

LP = 'Forecast LP v2, Holt-Winters (ours)'
LABELS = {BATT:'Battery Dijkstra',LEST:'LEST load-table Dijkstra',
          EXACT:'Coordinated Dijkstra, exact load',LOAD:'Load-only ablation',LP:'LP planner (saved reference)'}


def paired(a,b):
    x=np.asarray(a,float)-np.asarray(b,float)
    nz=x[np.abs(x)>1e-9]
    return {'mean_difference':float(x.mean()),'median_difference':float(np.median(x)),
            'W_T_L':[int((x>1e-9).sum()),int((np.abs(x)<=1e-9).sum()),int((x< -1e-9).sum())],
            'unadjusted_wilcoxon_p':float(wilcoxon(nz).pvalue) if len(nz) else 1.}


def main():
    old=json.loads(Path('results_dynamic.json').read_text())
    new=json.loads((OUT/'results_dynamic_corrected.json').read_text())
    arr=lambda d,m,sc:np.array([d[s][str(k)][m] for s in sc for k in range(30)])
    summary={'synthetic':{},'Intel':{}}
    lines=['# Corrected LEST load-booking rerun','',
           'All original parameters are fixed; no retuning. Synthetic: 30 deployments per scenario, R/N/B/D. Intel: 27 original start windows at each battery setting. Battery Dijkstra was rerun and matched every saved baseline. LP results and bounds are unchanged saved references.','',
           '## Synthetic traffic','',
           '| Method | Old R/N/B mean | Corrected R/N/B mean | Corrected D | W/T/L vs battery, all 120 cases |',
           '|---|---:|---:|---:|---|']
    for m in [BATT,LEST,EXACT,LOAD,LP]:
        before,after=arr(old,m,'RNB'),arr(new,m,'RNB')
        comp=paired(arr(new,m,'RNBD'),arr(new,BATT,'RNBD'))
        summary['synthetic'][m]={'old_RNB_mean':float(before.mean()),'new_RNB_mean':float(after.mean()),
                                 'new_D_mean':float(arr(new,m,'D').mean()),'all_120_vs_battery':comp,
                                 'per_scenario_means':{s:float(arr(new,m,s).mean()) for s in 'RNBD'}}
        lines.append(f"| {LABELS[m]} | {before.mean():.3f} | {after.mean():.3f} | {arr(new,m,'D').mean():.3f} | {'/'.join(map(str,comp['W_T_L']))} |")
    diff=np.array([[new[s][str(k)][LEST]-new[s][str(k)][BATT] for s in 'RNB'] for k in range(30)]).mean(axis=1)
    rng=np.random.default_rng(32)
    ci=np.quantile(diff[rng.integers(0,30,size=(20000,30))].mean(axis=1),[.025,.975])
    topology=paired(diff,np.zeros(30));topology['bootstrap_95CI']=ci.tolist()
    summary['topology_level_RNB_LEST_vs_battery']=topology
    gain=arr(new,LEST,'RNB').mean()-arr(new,BATT,'RNB').mean()
    lp_gap=arr(new,LP,'RNB').mean()-arr(new,BATT,'RNB').mean()
    summary['fraction_of_battery_to_LP_gap_closed']=float(gain/lp_gap)
    lines += ['',f"LEST gain over battery Dijkstra: {gain:.3f} percentage points. Fraction of the battery-to-LP-planner gap closed: {gain/lp_gap*100:.2f}%.",
              '',f"Deployment-level inference (30 topologies, averaging R/N/B within each topology): bootstrap 95% interval [{ci[0]:.3f}, {ci[1]:.3f}] points; unadjusted paired Wilcoxon p={topology['unadjusted_wilcoxon_p']:.3g}."]
    for e0,oldfile in [(1.,'results_intel_e1.json'),(1.5,'results_intel.json')]:
        a=json.loads(Path(oldfile).read_text());b=json.loads((OUT/f'results_intel_e{e0:g}_corrected.json').read_text())
        rows={};lines += ['',f'## Intel trace, E0 = {e0:g} J','',
                         '| Method | Old mean | Corrected mean | W/T/L vs battery | Censored |',
                         '|---|---:|---:|---|---:|']
        for m in ['Battery Dijkstra',LEST,EXACT,LP]:
            v=np.asarray(b['L'][m]);comp=paired(v,b['L']['Battery Dijkstra'])
            rows[m]={'old_mean':float(np.mean(a['L'][m])),'new_mean':float(v.mean()),'vs_battery':comp,'censored':b['censored'][m]}
            lines.append(f"| {LABELS.get(m,m)} | {np.mean(a['L'][m]):.3f} | {v.mean():.3f} | {'/'.join(map(str,comp['W_T_L']))} | {b['censored'][m]} |")
        summary['Intel'][str(e0)]=rows
    lines += ['', 'Intel results retain the original scoring formula for direct before/after comparison. Overlapping windows and censored runs prevent treating these 27 starts as independent uncensored deployments. Raw files explicitly distinguish failure from trace termination.',
              '', 'The fix books an already-routed source\'s own rate along its fixed path. Tests verify that final booked loads equal the rates summed over each subtree, including assigned relays, dead nodes, zero rates, both processing orders, and the original LEST deployments.',
              '', 'These results supersede the old LEST/exact-load/load-only numbers for these experiments. The manuscript\'s fixed 90.5%/90.8% text and affected tables/figures need updating. This run does not retune parameters or change the tier trigger, overhead model, radio assumptions, LP planner, or traffic.']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','old_RNB_mean','new_RNB_mean','new_D_mean','wins','ties','losses'])
        for m,r in summary['synthetic'].items():w.writerow([m,r['old_RNB_mean'],r['new_RNB_mean'],r['new_D_mean'],*r['all_120_vs_battery']['W_T_L']])
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
