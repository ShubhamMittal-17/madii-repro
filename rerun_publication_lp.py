"""Re-observe the main LP comparator, including event flags and fresh oracle bounds."""
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import env as E, intel as I, traffic as TR
from lp_oracle import oracle_T
from predictive import ForecastLP
OUT=Path('runs/publication_revision')

def job(task):
 kind,sc,seed=task
 if kind=='synthetic':
  w=E.WSN(seed=seed);h,tr=TR.generate(sc,w.pos[:w.n],seed+1000);gamma=.1
  make=lambda:E.WSN(seed=seed,traffic=tr)
 else:
  full=np.load('data/intel_lab/traffic_dT0.5_dL100.npy');h,tr=full[:seed],full[seed:];gamma=.3
  make=lambda:I.IntelWSN(e0=float(sc),traffic=tr)
 w=make();bound=oracle_T(w,tr)
 w=make();pol=ForecastLP(w,h,resolve_every=1,mu=1e-3,alpha=.05,beta=0.,gamma=gamma)
 failure=None
 while w.round<len(tr):
  if w.step(pol(w))['n_dead']>=1:failure=w.round;break
 rounds=w.round if failure is None else failure-1
 return dict(kind=kind,scenario=sc,seed=seed,oracle=bound,failure_round=failure,observed_rounds=w.round,censored=failure is None,completed_rounds=rounds,pct_oracle=100*rounds/bound)

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 jobs=[('synthetic',sc,s) for sc in 'RNBD' for s in range(30)]+[('intel',str(e),s) for e in [1.,1.5] for s in range(336,545,8)]
 previous=[]
 pth=OUT/'lp_fresh.jsonl'
 if pth.exists():previous=[json.loads(x) for x in pth.read_text().splitlines()]
 done={(x['kind'],x['scenario'],x['seed']) for x in previous};jobs=[j for j in jobs if j not in done]
 with Pool(2) as pool,pth.open('a') as log:
  for result in pool.imap_unordered(job,jobs):
   log.write(json.dumps(result)+'\n');log.flush();print(json.dumps({'completed':len(done)+1,'total':174,**result}),flush=True);done.add((result['kind'],result['scenario'],result['seed']))
 (OUT/'lp_complete.json').write_text(json.dumps({'completed':len(done)},indent=2)+'\n')
