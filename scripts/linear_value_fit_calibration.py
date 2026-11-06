"""Descriptive before/after calibration on diagnostic bank, not selection criterion."""
import copy,gzip,json
import numpy as np
from blokus.engine import Move,play
from blokus.linear_rl import prefix
from blokus.linear_value import Agent
from scripts.linear_value_diagnose import metrics,unpack
from scripts.linear_value_fit_eval import R,source

def main():
 with gzip.open(R/'diagnostic.json.gz','rt') as f:d=json.load(f)
 results={};arms=json.loads((R/'fixed-plan.json').read_text())['arms']
 for label in ['original']+arms:
  models={s:Agent.load(source(label,s)) for s in (3101,3102,3103)};visited=[];leaves=[];losses={k:[] for k in ('self','gen3')}
  for g in d['games']:
   s=prefix(g['prefix_seed']);a=models[g['seed']]
   for row in g['moves']:
    if s.turn==g['seat']:
     mean=sum(x.bit_count() for x in s.remaining)/2;phase=0 if mean>14 else 1 if mean>7 else 2;visited.append(dict(prediction=a.value(s),reward=g['reward'],phase=phase,distribution=g['distribution']))
    play(s,Move(*row) if row else None,validate=False)
  for root in d['roots']:
   a=models[root['seed']];predicted=[];targets=[]
   for c in root['candidates']:
    if c['rank']>=6:continue
    preds=[]
    for leaf in c['leaves']:
     s=unpack(leaf['state']);pred=a.value(s);sign=1 if s.turn==root['state']['turn'] else -1;preds.append(sign*pred);mean=sum(x.bit_count() for x in s.remaining)/2;phase=0 if mean>14 else 1 if mean>7 else 2
     for rr in leaf['rollouts']:leaves.append(dict(prediction=pred,reward=rr['reward'],phase=phase,distribution=root['distribution']))
    predicted.append(min(preds));targets.append(c['target'])
   chosen=int(np.argmax(predicted));losses[root['distribution']].append(max(targets)-targets[chosen])
  results[label]={dist:dict(visited=metrics([x for x in visited if x['distribution']==dist]),leaves=metrics([x for x in leaves if x['distribution']==dist]),mean_selection_loss=float(np.mean(losses[dist]))) for dist in ('self','gen3')}
 (R/'calibration-after.json').write_text(json.dumps(results,indent=2)+'\n');print('CALIBRATION COMPLETE')
if __name__=='__main__':main()
