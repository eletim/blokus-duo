import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts import linear_value_eval as base
from scripts.linear_hybrid_run import cluster
from blokus.linear_value import Agent,Search
from blokus.linear_rl import prefix
from blokus.linear_train2 import state_hash
R=Path('artifacts/linear-value-retrain');SEEDS=(3101,3102,3103)
def write(n,d):(R/n).write_text(json.dumps(d,indent=2)+'\n')
def source(label,seed):return Path(f'artifacts/linear-hybrid/exports/seed-{seed}.json') if label=='original' else R/f'{label}-{seed}/policy.json'
def model_sha(label,seed):return base.sha(f'artifacts/ppo-gen3/seed-{1751401+seed-3101}.pt') if label=='gen3' else base.sha(source(label,seed))
def make(label,seed):
 if label=='gen3':
  from blokus.ppo_policy import PPOAgent
  return PPOAgent(model_seed=1751401+seed-3101,device='cpu',backend='native')
 return Search(Agent.load(source(label,seed)),ms=6)
def setup():
 import torch
 torch.set_num_threads(1)
 for x in json.loads((R/'models.json').read_text()):assert base.sha(x['path'])==x['sha256']
 for arm in json.loads((R/'fixed-plan.json').read_text())['arms']:
  for seed in SEEDS:assert np.array_equal(Agent.load(source(arm,seed)).weights,Agent.load(source('original',seed)).weights)
 base.R=R;base.make=make;base.model_sha=model_sha

def bank(phase):
 p=R/f'{phase}-prefixes.json'
 if p.exists():return json.loads(p.read_text())
 a=np.load('/tmp/linear-retrain-training.npy',mmap_mode='r');assert hashlib.sha256(a.tobytes()).hexdigest()==json.loads((R/'training-summary.json').read_text())['union_sha256'];forbidden=set(json.loads((R/'exclusions.json').read_text())['states'])
 if phase=='confirmation':forbidden.update(x['hash'] for x in json.loads((R/'validation-prefixes.json').read_text())['prefixes'])
 rows=[];i=0;count=64 if phase=='validation' else 128;start=2300000000 if phase=='validation' else 2310000000
 while len(rows)<count:
  seed=start+i;i+=1;h=state_hash(prefix(seed))
  if h in forbidden or base.contains(a,h):continue
  forbidden.add(h);rows.append(dict(seed=seed,hash=h))
 d=dict(prefixes=rows,count=count,rejected=i-count,sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),training_sha256=hashlib.sha256(a.tobytes()).hexdigest());write(p.name,d);return d

def evaluate(phase):
 setup();arms=json.loads((R/'fixed-plan.json').read_text())['arms'];prefixes=bank(phase)
 if phase=='validation':pairs=[('original','gen3')]+[(a,b) for a in arms for b in ('original','gen3')]
 else:
  selection=json.loads((R/'selection.json').read_text());assert selection['selected'] is not None
  assert all(model_sha(label,int(seed))==h for label,hashes in selection['hashes'].items() for seed,h in hashes.items());a=selection['selected'];pairs=[(a,'original'),(a,'gen3')]
 summary={};allmatches={}
 for a,b in pairs:
  matches=[base.match(phase,a,b,s,prefixes) for s in SEEDS];summary[f'{a}/{b}']=cluster(matches);allmatches[f'{a}/{b}']=matches;write(f'{phase}-summary.json',summary)
 if phase=='validation':
  choices=[];gate={}
  for a in arms:
   stats=sum([d['left_search'] for d in allmatches[f'{a}/original']],[]);fallback=float(np.mean([x['fallback'] for x in stats]));passed=summary[f'{a}/original']['point']>=.48 and summary[f'{a}/gen3']['point']>=summary['original/gen3']['point']-.03 and fallback<.1;gate[a]=dict(pass_=passed,fallback=fallback)
   if passed:choices.append((summary[f'{a}/gen3']['point'],summary[f'{a}/original']['point'],a=='P',a))
  selected=max(choices)[3] if choices else None;write('selection.json',dict(selected=selected,gate=gate,hashes={a:{str(s):model_sha(a,s) for s in SEEDS} for a in ['original','gen3']+arms},seed_selection=False,checkpoint='fixed endpoint'));print('SELECTED',selected,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['validation','confirmation']);a=p.parse_args();evaluate(a.phase)
