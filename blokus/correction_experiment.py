"""Preregistered dev shortlist, new paired validation, gated fresh holdout."""
import argparse,json,time,random
import numpy as np
from .correction import CorrectionAgent
from .correction_data import ROOT
from .aligned_data import BASE
from .value_search import BudgetAgent
from .root_experiment import pooled
from . import root_experiment as harness
from .tournament import make_prefixes
from .value_experiment import write
from .research import sha
from .ai import seed_all
from .engine import State,play
SEEDS=[1001,1002,1003]

def agent(spec):
 if spec['method']=='old':return BudgetAgent('value',BASE,.5,24,spec['budget_ms'],'cuda')
 return CorrectionAgent(BASE,spec['budget_ms'],spec['policy'],alpha=spec['alpha'])

def match(left,right,fixture):
 old=harness.agent;harness.agent=agent
 try:return harness.match(left,right,fixture)
 finally:harness.agent=old

def spec(seed,step,alpha,budget):return dict(method='correction',seed=seed,step=step,alpha=alpha,budget_ms=budget,policy=str(ROOT/f'seed-{seed}/step-{step}.pt'))

def validation():
 scores=[]
 for step in [1000,3000]:
  for alpha in [.25,.5,1.]:
   ds=[next(r for r in json.loads((ROOT/f'seed-{seed}/metrics.json').read_text()) if r['step']==step)['dev'][str(alpha)] for seed in SEEDS]
   scores.append(dict(step=step,alpha=alpha,regret=float(np.mean([d['regret'] for d in ds])),zero_regret=float(np.mean([d['zero_regret'] for d in ds])),mse=float(np.mean([d['mse'] for d in ds])),zero_mse=float(np.mean([d['zero_mse'] for d in ds]))))
 shortlist=sorted(scores,key=lambda x:(x['regret'],x['alpha'],x['step']))[:2];write(ROOT/'dev-shortlist.json',dict(all=scores,selected=shortlist))
 fixture=make_prefixes(1021000,16);write(ROOT/'validation-prefixes.json',fixture);results=[];criteria=[]
 for config in shortlist:
  for budget in [6,25]:
   rows=[]
   for seed in SEEDS:
    r=match(spec(seed,config['step'],config['alpha'],budget),dict(method='old',budget_ms=budget),fixture);rows.append(r);results.append(r);write(ROOT/'validation-results.json',results);print('validation',seed,config,budget,r['summary'],flush=True)
   criteria.append(dict(**config,budget=budget,score=pooled(rows)['mean_difference']))
 choice=sorted(criteria,key=lambda x:(-x['score'],x['budget'],x['alpha'],x['step']))[0];budget=choice['budget'];zero_rows=[]
 for seed in SEEDS:
  left=spec(seed,choice['step'],choice['alpha'],budget);right=dict(**{**left,'method':'zero','alpha':0.})
  r=match(left,right,fixture);zero_rows.append(r);results.append(r);write(ROOT/'validation-results.json',results);print('zero',seed,r['summary'],flush=True)
 rows=[r for r in results if r['right']['spec']['method']=='old' and r['left']['spec']['step']==choice['step'] and r['left']['spec']['alpha']==choice['alpha'] and r['left']['spec']['budget_ms']==budget]
 ms=[d['policy_ms'] for r in rows for g in r['games'] for d in g['decisions'] if d['who']==0];old=pooled(rows);zero=pooled(zero_rows)
 gate=dict(positive_old=old['mean_difference']>0,each_seed_above_minus2=all(r['summary']['mean_difference']>-2 for r in rows),positive_zero=zero['mean_difference']>0,inference_below_1ms=np.mean(ms)<1,dev_regret_improved=choice['regret']<choice['zero_regret'])
 selection=dict(choice=choice,criteria=criteria,seeds=SEEDS,old=old,zero=zero,mean_inference_ms=float(np.mean(ms)),gate={k:bool(v) for k,v in gate.items()},advance=bool(all(gate.values())),prefix_sha256=fixture['sha256'],checkpoints={str(seed):dict(path=spec(seed,choice['step'],choice['alpha'],budget)['policy'],sha256=sha(spec(seed,choice['step'],choice['alpha'],budget)['policy'])) for seed in SEEDS})
 write(ROOT/'selection.json',selection);print('SELECTION',selection,flush=True)

def holdout():
 sel=json.loads((ROOT/'selection.json').read_text());assert sel['advance'],'preregistered gate failed';c=sel['choice'];fixture=make_prefixes(1022000,96);write(ROOT/'holdout-prefixes.json',fixture);results=[]
 for seed in SEEDS:
  left=spec(seed,c['step'],c['alpha'],c['budget']);assert sha(left['policy'])==sel['checkpoints'][str(seed)]['sha256']
  for right in [dict(method='old',budget_ms=c['budget']),dict(**{**left,'method':'zero','alpha':0.})]:
   r=match(left,right,fixture);results.append(r);write(ROOT/'holdout-results.json',results);print('holdout',seed,right['method'],r['summary'],flush=True)
 write(ROOT/'holdout-pooled.json',{m:pooled([r for r in results if r['right']['spec']['method']==m]) for m in ['old','zero']})

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['validation','holdout','play']);p.add_argument('--seed',type=int,default=1001);a=p.parse_args();seed_all(1000)
 if a.command=='validation':return validation()
 if a.command=='holdout':return holdout()
 c=json.loads((ROOT/'selection.json').read_text())['choice'];bots=[agent(spec(a.seed,c['step'],c['alpha'],c['budget'])),agent(dict(method='old',budget_ms=c['budget']))];s=State();rng=random.Random(1050000)
 while not s.done:play(s,bots[s.turn].act(s,rng),validate=True)
 print(dict(scores=[s.score(0),s.score(1)],student=bots[0].stats(),old=bots[1].stats()))
if __name__=='__main__':main()
