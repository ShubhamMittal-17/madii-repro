"""Evaluate three held-out-selected corrected reconstructions on dynamic/Intel traffic.
Training seeds and inference repeats are averaged within each deployment, never treated
as additional independent topologies. Raw event/censor observations are retained.
"""
import hashlib,json
from pathlib import Path
import numpy as np,torch
import env as E,intel as I,traffic as TR
from evaluate import load
from train import greedy_action
OUT=Path('runs/publication_revision')

def main():
 torch.set_num_threads(1)
 bounds=json.loads(Path('runs/correct_load_booking/results_dynamic_corrected.json').read_text())
 intelbounds={e:json.loads(Path(f'runs/correct_load_booking/results_intel_e{e:g}_corrected.json').read_text()) for e in [1.,1.5]}
 result={'inference_rule':'1000000 + 100000*repeat + deployment seed/start','training_seeds':[0,1,2],'inference_repeats':3,'models':{}}
 full=np.load('data/intel_lab/traffic_dT0.5_dL100.npy')
 for trainingseed in range(3):
  path=Path(f'checkpoints/madii_correct_reward_seed{trainingseed}_best.pt');net,cfg=load(str(path));ck=torch.load(path,map_location='cpu',weights_only=False)
  model={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected_episode':ck['ep'],'training_args':cfg,'synthetic':{},'intel':{}}
  for sc in 'RNBD':
   model['synthetic'][sc]={}
   for seed in range(30):
    w=E.WSN(seed=seed);_,tr=TR.generate(sc,w.pos[:w.n],seed+1000);oracle=bounds[sc][str(seed)]['oracle'];rows=[]
    for repeat in range(3):
     torch.manual_seed(1000000+100000*repeat+seed);w=E.WSN(seed=seed,traffic=tr);failed=None
     while w.round<len(tr):
      if w.step(greedy_action(net,w))['n_dead']>=1:failed=w.round;break
     completed=w.round if failed is None else failed-1
     rows.append(dict(failure_round=failed,observed_rounds=w.round,censored=failed is None,pct_oracle=100*completed/oracle))
    model['synthetic'][sc][str(seed)]=rows
   print(json.dumps({'training_seed':trainingseed,'scenario':sc,'complete':30}),flush=True)
  for energy in [1.,1.5]:
   model['intel'][str(energy)]={}
   for start in range(336,545,8):
    tr=full[start:];oracle=intelbounds[energy]['oracle'][str(start)];rows=[]
    for repeat in range(3):
     torch.manual_seed(1000000+100000*repeat+start);w=I.IntelWSN(e0=energy,traffic=tr);failed=None
     while w.round<len(tr):
      if w.step(greedy_action(net,w))['n_dead']>=1:failed=w.round;break
     completed=w.round if failed is None else failed-1
     rows.append(dict(failure_round=failed,observed_rounds=w.round,censored=failed is None,pct_oracle=100*completed/oracle))
    model['intel'][str(energy)][str(start)]=rows
   print(json.dumps({'training_seed':trainingseed,'energy':energy,'complete':27}),flush=True)
  result['models'][str(trainingseed)]=model
  (OUT/'madii_dynamic_partial.json').write_text(json.dumps(result,indent=2)+'\n')
 (OUT/'madii_dynamic.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
