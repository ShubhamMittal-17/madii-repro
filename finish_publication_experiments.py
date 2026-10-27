"""Wait for the two requested training runs, then evaluate selected and final checkpoints."""
import json,subprocess,time
from pathlib import Path
import torch
OUT=Path('runs/publication_revision');PYTHON='/workspace/scratch/8dda305de2a7/madii_venv/bin/python'
while True:
 ready=[]
 for seed in [1,2]:
  log=Path(f'logs/madii_correct_reward_seed{seed}.jsonl')
  lines=log.read_text().splitlines() if log.exists() else []
  evaluated=any(json.loads(x).get('ep')==1000 and 'eval' in json.loads(x) for x in lines)
  try:
   saved=torch.load(f'checkpoints/madii_correct_reward_seed{seed}.pt',map_location='cpu',weights_only=False)['ep']==1000
  except (OSError,RuntimeError,EOFError):
   saved=False
  ready.append(evaluated and saved)
 if all(ready):break
 time.sleep(10)
checkpoints=[f'checkpoints/madii_correct_reward_seed{s}{suffix}.pt' for s in range(3) for suffix in ['_best','']]
subprocess.run([PYTHON,'evaluate_correct_reward.py','--checkpoints',*checkpoints,'--threads','1','--output',str(OUT/'madii_static.json')],check=True)
subprocess.run([PYTHON,'evaluate_publication_madii.py'],check=True)
(OUT/'madii_complete.json').write_text(json.dumps({'training_seeds':[0,1,2],'episodes_per_seed':1000,'inference_repeats':3},indent=2)+'\n')
