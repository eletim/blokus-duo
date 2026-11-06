import gzip,json
from pathlib import Path
import numpy as np
from blokus.engine import Move,play
from blokus.linear_rl import prefix
from blokus.linear_train2 import state_hash
from scripts.linear_value_eval import contains
from scripts.linear_value_diagnose import unpack
R=Path('artifacts/linear-value-retrain')
def main():
 with gzip.open(R/'diagnostic.json.gz','rt') as f:d=json.load(f)
 seen=set();starts=set()
 for g in d['games']:
  s=prefix(g['prefix_seed']);starts.add(state_hash(s))
  for row in g['moves']:seen.add(state_hash(s));play(s,Move(*row) if row else None,validate=False)
  seen.add(state_hash(s))
 for root in d['roots']:
  for c in root['candidates']:
   for leaf in c['leaves']:
    for roll in leaf['rollouts']:
     s=unpack(leaf['state'])
     for row in roll['moves']:seen.add(state_hash(s));play(s,Move(*row) if row else None,validate=False)
     seen.add(state_hash(s))
 old=set(json.loads(Path('artifacts/linear-value/exclusions.json').read_text())['states'])
 for folder in ('linear-value','linear-hybrid'):
  for phase in ('validation','confirmation'):old.update(x['hash'] for x in json.loads(Path(f'artifacts/{folder}/{phase}-prefixes.json').read_text())['prefixes'])
 assert not old&starts
 a=np.load('/tmp/linear-retrain-training.npy',mmap_mode='r');overlap=sum(contains(a,h) for h in seen);assert overlap==0
 train=json.loads((R/'training-audit.json').read_text());n=json.loads((R/'fixed-plan.json').read_text())['games_per_seed_arm']
 for row in train['rows']:assert row['balance']=={'fixed/0':n*3//8,'fixed/1':n*3//8,'initial/0':n//8,'initial/1':n//8}
 result=dict(pass_=True,diagnostic_unique_states=len(seen),diagnostic_training_overlap=int(overlap),old_eval_starts_overlap=0,opponent_seat_balance=True)
 if (R/'validation-prefixes.json').exists():
  forbidden=set(json.loads((R/'exclusions.json').read_text())['states']);banks={}
  for phase in ('validation','confirmation'):
   path=R/f'{phase}-prefixes.json'
   if not path.exists():continue
   banks[phase]={x['hash'] for x in json.loads(path.read_text())['prefixes']};assert not banks[phase]&forbidden;assert not any(contains(a,h) for h in banks[phase])
  if 'confirmation' in banks:assert not banks['confirmation']&banks['validation']
  result['evaluation_starts_disjoint']=True
 (R/'separation-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
