"""Generate the revised main evidence exclusively from completed corrected experiments.
Run from repo root after publication_revision experiment gates are complete.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/publication_revision';BODY=ROOT/'paper/body';FIG=ROOT/'paper/figs'
J=lambda p:json.loads((ROOT/p).read_text())
LEST='Load-aware Dijkstra, LEST load table (4 tiers)';EXACT='Load-aware Dijkstra (ours, no LP)';BATT='Battery Dijkstra (reactive)';LOAD='Load table only (ablation, no battery)';LP='Forecast LP v2, Holt-Winters (ours)';MAD='MADII-inspired, corrected (3 training seeds)'
LABEL={LEST:r'\textbf{LEST load-table Dijkstra}',EXACT:'Coordinated Dijkstra, exact load',BATT:'Battery-weighted Dijkstra',LOAD:'Load only (no battery weighting)',LP:'LP v2, largest share each round',MAD:'MADII-inspired, corrected (3 seeds)'}

def paired(a,b):
 d=np.asarray(a)-np.asarray(b);nz=d[np.abs(d)>1e-9];ranks=rankdata(np.abs(nz));p=float(wilcoxon(nz).pvalue) if len(nz) else 1.
 return dict(n=len(d),mean=float(d.mean()),median=float(np.median(d)),W=int((d>1e-9).sum()),T=int((abs(d)<=1e-9).sum()),L=int((d< -1e-9).sum()),p=p,r=float((ranks[nz>0].sum()-ranks[nz<0].sum())/ranks.sum()) if len(nz) else 0.)
def holm(rows):
 prev=0.
 for k,i in enumerate(sorted(range(len(rows)),key=lambda i:rows[i]['p'])):
  prev=max(prev,min(1.,(len(rows)-k)*rows[i]['p']));rows[i]['p_holm']=prev
 return rows
def pt(p):
 if p>=.001:return f'{p:.3f}'
 m,e=f'{p:.1e}'.split('e');return m+r'\times10^{'+str(int(e))+'}'
def wtl(x):return f"{x['W']}/{x['T']}/{x['L']}"
def latex_table(caption,label,cols,header,rows,wide=True):
 env='widetab' if wide else 'table'
 return '\n'.join([r'\begin{'+env+'}',r'\centering',r'\caption{'+caption+'}'+r'\label{'+label+'}',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{'+cols+'}',r'\toprule',header+r'\\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',r'\end{'+env+'}'])

def main():
 for gate in ['madii_complete.json','lp_complete.json']:
  if not (OUT/gate).exists():raise SystemExit(f'Required completed experiment missing: {gate}')
 static=J('runs/publication_revision/madii_static.json');madii=J('runs/publication_revision/madii_dynamic.json')
 dyn=J('runs/correct_load_booking/results_dynamic_corrected.json');lp=[json.loads(x) for x in (OUT/'lp_fresh.jsonl').read_text().splitlines()];assert len(lp)==174
 intel={e:J(f'runs/correct_load_booking/results_intel_e{e:g}_corrected.json') for e in [1.,1.5]};iraw=J('runs/correct_load_booking/intel_raw.json');starts=list(range(336,545,8))
 for row in lp:
  if row['kind']=='synthetic':
   old=dyn[row['scenario']][str(row['seed'])]['oracle'];assert old==row['oracle'],('oracle mismatch',row)
   dyn[row['scenario']][str(row['seed'])][LP]=row['pct_oracle']
  else:
   e=float(row['scenario']);assert intel[e]['oracle'][str(row['seed'])]==row['oracle'];intel[e]['L'][LP][starts.index(row['seed'])]=row['pct_oracle']
 for sc in 'RNBD':
  for seed in range(30):dyn[sc][str(seed)][MAD]=float(np.mean([x['pct_oracle'] for m in madii['models'].values() for x in m['synthetic'][sc][str(seed)]]))
 for e,d in intel.items():
  for m in ['Battery Dijkstra',LEST,EXACT]:
   d['L'][m]=[100*(iraw[str(e)][str(s)]['methods'][m]['observed_rounds'] if iraw[str(e)][str(s)]['methods'][m]['censored'] else iraw[str(e)][str(s)]['methods'][m]['failure_round']-1)/d['oracle'][str(s)] for s in starts]
  d['L'][MAD]=[float(np.mean([x['pct_oracle'] for m in madii['models'].values() for x in m['intel'][str(e)][str(s)]])) for s in starts]
  d['censored'][LP]=sum(x['censored'] for x in lp if x['kind']=='intel' and float(x['scenario'])==e)
  d['censored'][MAD]=sum(x['censored'] for m in madii['models'].values() for v in m['intel'][str(e)].values() for x in v)
  d['score_definition']='Observed completed death-free rounds / horizon-truncated oracle. Censoring retained; no inference across overlapping windows.'
 core=[LEST,EXACT,LP,BATT,LOAD,MAD]
 # Keep archived exploratory result methods out of the revised main comparison.
 core_dyn={sc:{s:{m:v for m,v in row.items() if m in core or m=='oracle'} for s,row in rows.items()} for sc,rows in dyn.items()}
 (OUT/'synthetic_core.json').write_text(json.dumps(core_dyn,indent=2)+'\n')
 (ROOT/'results_dynamic.json').write_text(json.dumps(core_dyn,indent=2)+'\n')
 (ROOT/'results_dynamic.md').write_text('Corrected main evidence is in runs/publication_revision/synthetic_core.json and paper/body/publication_results.tex. Original exploratory rows are archived under runs/publication_revision/original_results_dynamic.json.\n')
 for e,d in intel.items():
  d['L']={m:d['L'][m] for m in [LEST,EXACT,LP,'Battery Dijkstra',MAD]};(OUT/f'intel_core_e{e:g}.json').write_text(json.dumps(d,indent=2)+'\n')
  (ROOT/('results_intel_e1.json' if e==1. else 'results_intel.json')).write_text(json.dumps(d,indent=2)+'\n')
 per=lambda m,sc:np.array([dyn[sc][str(s)][m] for s in range(30)])
 topology=lambda m:np.mean([per(m,sc) for sc in 'RNB'],axis=0)
 allcases=lambda m:np.concatenate([per(m,sc) for sc in 'RNBD'])
 pairs=[(LEST,BATT),(LP,LEST),(EXACT,LEST),(LEST,LOAD),(MAD,BATT)]
 tests=holm([dict(method_a=a,method_b=b,**paired(topology(a),topology(b))) for a,b in pairs])
 delta=topology(LEST)-topology(BATT);rng=np.random.default_rng(32);ci=np.quantile(delta[rng.integers(0,30,(20000,30))].mean(axis=1),[.025,.975])
 gain=float(delta.mean());gap=float(topology(LP).mean()-topology(BATT).mean());fraction=100*gain/gap
 stats=dict(unit='topology: mean of R/N/B, n=30',family=tests,bootstrap=dict(seed=32,resamples=20000,mean_gain= gain,CI95=ci.tolist()),gap_closed_pct=fraction)
 (OUT/'statistics_core.json').write_text(json.dumps(stats,indent=2)+'\n')
 (ROOT/'results_wilcoxon.json').write_text(json.dumps(stats,indent=2)+'\n')
 (ROOT/'results_wilcoxon.md').write_text('Corrected primary inference uses 30 R/N/B topology averages. See results_wilcoxon.json and paper/body/publication_results.tex for the five-comparison Holm family. Intel overlapping windows are descriptive.\n')
 selected=[static['models'][f'madii_correct_reward_seed{s}_best'] for s in range(3)]
 final=[static['models'][f'madii_correct_reward_seed{s}'] for s in range(3)]
 meanf=lambda rows,key:float(np.mean([x[key] for x in rows]))
 static_fnd=meanf(selected,'mean_FND');static_pct=meanf(selected,'mean_pct_ceiling')
 sections=[r'\subsection{Static traffic and corrected neural reconstruction}\label{sec:static}',
  f"The corrected reconstruction is evaluated with three independently trained seeds, validation-only checkpoint selection and three fixed inference repeats per deployment. Selected checkpoints average {static_fnd:.2f} rounds to FND and {static_pct:.2f}\\% of the fractional-flow ceiling. The three seed means range from {min(x['mean_FND'] for x in selected):.2f} to {max(x['mean_FND'] for x in selected):.2f} rounds. Final checkpoints average {meanf(final,'mean_FND'):.2f} rounds. These are measurements of the reconstruction; MADII's printed FND of 19 rounds~\\cite{{yang2025madii}} is an external result from its own experiments."]
 sections.append('Under the primary fixed inference stream, all three selected models send every source directly to the sink in the first round of all 30 static test topologies. This observed behavior helps explain the short FND; it is not a result for the published authors implementation.')
 rows=[]
 for s in range(3):
  a,b=selected[s],final[s];rows.append(f"{s} & {a['episode']} & {a['mean_FND']:.2f} & {a['mean_pct_ceiling']:.2f} & {a['mean_HND']:.2f} & {b['mean_FND']:.2f}\\\\")
 sections.append(latex_table('Corrected reconstruction by training seed. Each entry averages 30 held-out topologies and three fixed inference repeats. The selected episode uses validation seeds 500--504 only.','tab:repro',r'@{}rrrrrr@{}','Seed & Selected ep. & FND & Ceiling (\\%) & HND & Final FND',rows))
 classical=J('results_compare.json');strows=[]
 for label,key in [('Adaptive LP (every 8 rounds)','LP-flow adaptive (classical)'),('Battery-weighted Dijkstra','battery Dijkstra (classical)'),('RL warm-started Dijkstra','RL-Dijkstra warm-start [dijkstra_rl_v2]'),('Static LP (one solve)','LP-flow static (classical)'),('Minimum-energy Dijkstra','min-energy Dijkstra'),('FCM clustering','FCM clustering (paper baseline)')]:
  v=classical[key];strows.append(f"{label} & {np.mean(v['FND']):.1f} & {np.mean(v['pct_ceiling']):.1f} & {np.min(v['pct_ceiling']):.1f}\\\\")
 sections.append(latex_table('Static classical comparators, rerun on 30 held-out topologies. FND is the death event round; normalized scores count completed death-free rounds.','tab:static',r'@{}L{0.47\textwidth}rrr@{}','Method & FND & Ceiling (\\%) & Worst (\\%)',strows))
 sections += [r'\subsection{Time-varying traffic}\label{sec:dynamic}',f"On the R/N/B score, LEST reaches {topology(LEST).mean():.1f}\\% versus {topology(BATT).mean():.1f}\\% for battery Dijkstra and {topology(LP).mean():.1f}\\% for the fresh LP comparator. It closes {fraction:.1f}\\% of the battery-to-planner gap. Across R/N/B/D the descriptive counts are {wtl(paired(allcases(LEST),allcases(BATT)))}. On the pattern shift D, LEST reaches {per(LEST,'D').mean():.1f}\\% versus {per(BATT,'D').mean():.1f}\\%. Exact-load coordination reaches {topology(EXACT).mean():.1f}\\%; load-only routing reaches {topology(LOAD).mean():.1f}\\%. These corrected ablations support combining battery and load. Earlier tier-trigger, granularity and tuning ablations used the pre-fix booking code and are not claimed as corrected evidence."]
 rows=[]
 for m in core:
  means=[per(m,s).mean() for s in 'RNBD'];rows.append(f"{LABEL[m]} & {means[0]:.1f} & {means[1]:.1f} & {means[2]:.1f} & {np.mean(means[:3]):.1f} & {means[3]:.1f} & {wtl(paired(allcases(m),allcases(BATT)))}\\\\")
 sections.append(latex_table('Corrected synthetic results. R: surges; N: constant; B: bursts; D: pattern shift. R/N/B reuses the same 30 topologies. W/T/L over all 120 cases is descriptive; it is not an inferential sample size. Neural entries average three training seeds and three inference streams within each case.','tab:dynamic',r'@{}L{0.38\textwidth}rrrrrl@{}','Method & R & N & B & R/N/B & D & W/T/L',rows))
 sections += [r'\subsection{Topology-level statistical analysis}\label{sec:stats}',f"The primary LEST-minus-battery gain is {gain:.2f} percentage points (95\\% whole-topology bootstrap interval [{ci[0]:.2f}, {ci[1]:.2f}]). The unadjusted two-sided Wilcoxon result is $p={pt(tests[0]['p'])}$; Holm adjustment across the five listed comparisons gives $p={pt(tests[0]['p_holm'])}$. Each test uses 30 paired R/N/B topology averages."]
 rows=[f"{LABEL[x['method_a']].replace(r'\textbf','')} vs {LABEL[x['method_b']].replace(r'\textbf','')} & {x['mean']:+.2f} & {x['median']:+.2f} & {wtl(x)} & ${pt(x['p'])}$ & ${pt(x['p_holm'])}$ & {x['r']:+.2f}\\\\" for x in tests]
 sections.append(latex_table('Two-sided Wilcoxon tests on 30 topology averages. Differences are percentage points; Holm adjustment covers exactly these five comparisons. W/T/L here counts topologies; $r$ is rank-biserial correlation.','tab:wilcoxon',r'@{}L{0.35\textwidth}rrlrrr@{}','Comparison & Mean & Median & W/T/L & $p$ & Holm $p$ & $r$',rows))
 sections += [r'\subsection{Intel trace windows and censoring}\label{sec:intel}',r'Intel results are observed lifetimes over 27 overlapping start windows on a single layout. A run reaching the end of the trace contributes its full observed horizon and is marked censored; equal observed durations do not establish equal eventual lifetimes. Scores use a horizon-truncated oracle. The windows are not independent replications and no confirmatory significance test is reported.']
 rows=[]
 for m in [LEST,EXACT,LP,'Battery Dijkstra',MAD]:
  label=LABEL.get(m,'Battery-weighted Dijkstra');parts=[]
  for e in [1.,1.5]:
   d=intel[e];v=np.array(d['L'][m]);c=d['censored'][m];parts += [f'{v.mean():.1f}',f'{v.min():.1f}',str(c) if m!=MAD else f'{c}/243']
  rows.append(label+' & '+' & '.join(parts)+r'\\')
 sections.append(latex_table('Observed Intel lifetime ratios and censor counts. Classical censor counts are out of 27 windows; neural counts are out of 243 trajectories (27 windows, three training seeds, three inference streams). These counts are not independent sample sizes.','tab:intel',r'@{}L{0.4\textwidth}rrrrrr@{}',r'Method & Mean 1 J & Worst & Cens. & Mean 1.5 J & Worst & Cens.',rows))
 sections += [r'\subsection{Application traffic and planned hops}\label{sec:computation}',r'The traffic experiment runs constant generation until half the nodes are dead. Its application-data volume excludes the LEST byte; its energy cost includes that byte. Planned hops count each tree link once and weight generated packets across rounds. Direct retry hops, MAC delay and actual transmission-event counts are not measured by this proxy.']
 tr=J('results_traffic.json');rows=[]
 assert len(tr)==30 and all('MADII (corrected reconstruction, seed 0)' in row for row in tr.values()), 'Corrected traffic rerun has not completed'
 for label,key in [('LP planner v2','Planner v2 (LP every round, ours)'),('LEST Dijkstra',LEST),('Battery Dijkstra','Battery Dijkstra'),('Minimum-energy Dijkstra','Min-energy Dijkstra')]:
  vv=[tr[str(s)][key] for s in range(30)];avg=lambda f:np.mean([f(v) for v in vv]);rows.append(f"{label} & {avg(lambda v:v['at']['0.01']['rounds']):.1f} & {avg(lambda v:v['at']['0.5']['rounds']):.1f} & {avg(lambda v:v['at']['0.5']['DDV']):.0f} & {avg(lambda v:v['at']['0.5']['EE']):.0f} & {100*avg(lambda v:v['delivery']):.2f} & {avg(lambda v:v['hops']):.2f}\\\\")
 neural=[row['0.5'] for model in selected for repeat in model['raw'] for row in repeat]
 rows.append(f"Corrected neural (3 seeds) & {static_fnd:.1f} & {meanf(selected,'mean_HND'):.1f} & {np.mean([x['DDV'] for x in neural]):.0f} & {np.mean([x['EE'] for x in neural]):.0f} & -- & --\\\\")
 sections.append(latex_table('Corrected traffic metrics on 30 constant-traffic topologies. Neural DDV and EE use selected models with three inference repeats; neural delivery ratio and planned hops were not separately measured in this aggregate.','tab:traffic',r'@{}L{0.31\textwidth}rrrrrr@{}',r'Router & FND & HND & kbit to HND & kbit/J & Delivery (\%) & Hops',rows))
 (BODY/'publication_results.tex').write_text('\n\n'.join(sections)+'\n')
 # Figures use the same exported main data; paired synthetic points represent n=30.
 FIG.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(6.6,2.9));methods=[MAD,BATT,LEST,LP];vals=[topology(m).mean() for m in methods];ax.bar(range(4),vals,color=['#5676ad','#5c986d','#d99835','#9b619c']);ax.set_xticks(range(4),['Corrected neural\n3 training seeds','Battery\nDijkstra','LEST\nDijkstra','LP every\nround']);ax.set_ylabel('R/N/B mean, % of oracle');ax.set_ylim(0,105)
 for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.1f}',ha='center')
 fig.tight_layout();fig.savefig(FIG/'ladder.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(6.6,3.1));x=topology(BATT);y=topology(LEST)
 axes[0].scatter(x,y,s=22,color='#416ead');axes[0].plot([70,100],[70,100],'--',color='gray');axes[0].set_title('Synthetic: 30 topology averages');axes[0].set_xlabel('Battery Dijkstra, %');axes[0].set_ylabel('LEST Dijkstra, %')
 x=np.array(intel[1.]['L']['Battery Dijkstra']);y=np.array(intel[1.]['L'][LEST]);c=np.array([iraw['1.0'][str(s)]['methods'][LEST]['censored'] or iraw['1.0'][str(s)]['methods']['Battery Dijkstra']['censored'] for s in starts]);axes[1].scatter(x[~c],y[~c],s=22,color='#b7752d',label='Both deaths observed');axes[1].scatter(x[c],y[c],s=35,marker='^',facecolors='none',edgecolors='#b7752d',label='At least one censored');axes[1].plot([45,100],[45,100],'--',color='gray');axes[1].set_title('Intel: 27 overlapping windows');axes[1].set_xlabel('Battery observed ratio, %');axes[1].set_ylabel('LEST observed ratio, %');axes[1].legend(fontsize=6,loc='upper left');fig.tight_layout();fig.savefig(FIG/'paired.pdf');plt.close(fig)
 (BODY/'publication_results.tex').write_text((BODY/'publication_results.tex').read_text()+r'''
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figs/ladder.pdf}\caption{Main corrected synthetic comparison. Neural performance is averaged within each topology across training and inference seeds.}\label{fig:ladder}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figs/paired.pdf}\caption{Paired corrected results. The synthetic panel uses one R/N/B average per topology. Intel triangles mark pairs with at least one censored trajectory; those observations do not establish eventual lifetimes.}\label{fig:paired}\end{figure}
''')
 manifest={'generated_by':'paper/rebuild_publication.py','training_seeds':[0,1,2],'inference_repeats':3,'lp_fresh_cases':len(lp),'synthetic_topologies':30,'synthetic_cases':120,'Intel_windows_per_battery':27,'author_metadata':'pending confirmation','inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'madii_static.json',OUT/'madii_dynamic.json',OUT/'lp_fresh.jsonl',ROOT/'results_traffic.json',ROOT/'results_compare.json',ROOT/'runs/correct_load_booking/synthetic_raw.json',ROOT/'runs/correct_load_booking/intel_raw.json']}}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'selected_mean_FND':static_fnd,'selected_mean_pct':static_pct,'LEST':topology(LEST).mean(),'LP':topology(LP).mean(),'gap_closed_pct':fraction},indent=2))

if __name__=='__main__':main()
