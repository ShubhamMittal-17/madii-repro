"""Prepare corrected data and manuscript statistics, preserving original result files."""
import json, shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/publication_revision';OUT.mkdir(parents=True,exist_ok=True)
for name in ['results_dynamic.json','results_intel_e1.json','results_intel.json','results_eval.json']:
 p=OUT/('original_'+name)
 if not p.exists(): shutil.copy2(ROOT/name,p)
shutil.copy2(ROOT/'runs/correct_load_booking/results_dynamic_corrected.json',ROOT/'results_dynamic.json')
shutil.copy2(ROOT/'runs/correct_load_booking/results_intel_e1_corrected.json',ROOT/'results_intel_e1.json')
shutil.copy2(ROOT/'runs/correct_load_booking/results_intel_e1.5_corrected.json',ROOT/'results_intel.json')
# Use completed rounds if the trace ends without a death. Preserve the event flag.
raw=json.loads((ROOT/'runs/correct_load_booking/intel_raw.json').read_text())
for energy,filename in [('1.0','results_intel_e1.json'),('1.5','results_intel.json')]:
 d=json.loads((ROOT/filename).read_text());starts=sorted(d['oracle'],key=int)
 for m,values in d['L'].items():
  for i,start in enumerate(starts):
   obs=raw[energy][start]['methods'].get(m)
   if obs:
    values[i]=100*(obs['observed_rounds'] if obs['censored'] else obs['failure_round']-1)/d['oracle'][start]
  # Existing unaffected routers have only aggregate censor counts, not event flags;
  # do not silently "repair" their scores without re-observing their trajectories.
 d['score_definition']='Completed death-free rounds / horizon-truncated oracle. Corrected trajectory-level censoring only for BATT, LEST and exact-load; other methods retained legacy scoring.'
 d['event_observations']={m:[raw[energy][s]['methods'][m] for s in starts] for m in raw[energy][starts[0]]['methods']}
 (ROOT/filename).write_text(json.dumps(d,indent=2)+'\n')
print('Prepared corrected results; original inputs preserved')
