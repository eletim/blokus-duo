"""Only reachable after the registered game-generalization gate passes."""
import argparse,json
import numpy as np
from .scaling_data import ROOT
from .scaling_train import train
from .scaling_experiment import SEEDS,aggregate
from .correction_experiment import match
from .root_experiment import pooled
from .tournament import make_prefixes
from .value_experiment import write
from .research import sha

def spec(size,c,seed):return dict(method='correction',condition=c,seed=seed,step=1000,alpha=.5,budget_ms=25,policy=str(ROOT/f'n{size}-f-1-s{seed}-{c}/step-1000.pt'))

def run():
 decision=json.loads((ROOT/'decision.json').read_text());assert decision['advance'] and decision['final'],'learning gate failed or curve incomplete: no battle validation allowed';size=decision['size'];conditions=decision['accepted_conditions']
 for c in conditions:
  assert aggregate(size,c)['advance']
  for seed in SEEDS:train(size,-1,seed,c,steps=1000)
 fixture=make_prefixes(1121000,32);write(ROOT/'validation-prefixes.json',fixture);results=[];scores={}
 for c in conditions:
  rows=[]
  for seed in SEEDS:
   r=match(spec(size,c,seed),dict(method='old',budget_ms=25),fixture);rows.append(r);results.append(r);write(ROOT/'validation-results.json',results)
  scores[c]=pooled(rows)['mean_difference']
 c=max(sorted(scores),key=scores.get);zero=[]
 for seed in SEEDS:
  left=spec(size,c,seed);r=match(left,dict(**{**left,'method':'zero','alpha':0.}),fixture);zero.append(r);results.append(r);write(ROOT/'validation-results.json',results)
 selection=dict(size=size,condition=c,step=1000,alpha=.5,budget_ms=25,criteria=scores,zero=pooled(zero),advance=scores[c]>0 and pooled(zero)['mean_difference']>0,checkpoint_hashes={str(seed):sha(spec(size,c,seed)['policy']) for seed in SEEDS})
 write(ROOT/'battle-selection.json',selection)
 if not selection['advance']:return
 fixture=make_prefixes(1122000,96);write(ROOT/'holdout-prefixes.json',fixture);hold=[]
 for seed in SEEDS:
  left=spec(size,c,seed);assert sha(left['policy'])==selection['checkpoint_hashes'][str(seed)]
  for right in [dict(method='old',budget_ms=25),dict(**{**left,'method':'zero','alpha':0.})]:
   r=match(left,right,fixture);hold.append(r);write(ROOT/'holdout-results.json',hold)
 write(ROOT/'holdout-pooled.json',{m:pooled([r for r in hold if r['right']['spec']['method']==m]) for m in ['old','zero']})
if __name__=='__main__':run()
