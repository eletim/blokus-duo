import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,random,gzip
import numpy as np,torch
import gpu_generation as ref
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch,key
from blokus.ppo_frontier import FrontierSearch,FrontierEvaluator,run_frontier
from blokus.engine import State,Move,play
from pathlib import Path

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 results=[]
 for seed in ref.e.SEEDS:
  p=PPOAgent(seed,'cpu',backend='native');gp=PPOAgent(seed,'cuda',backend='native')
  d=json.load(gzip.open(ref.R/'cpu-1'/f'{seed}-0.json.gz','rt'));states=[];s=State()
  for i,m in enumerate(d['moves']):
   if i in (0,8,16,24,32,40):states.append(__import__('copy').deepcopy(s))
   play(s,Move(*m) if m else None)
  # Terminal is checked through value_only; policy act is undefined after done.
  for s in states:
   before=key(s);r1=random.Random(77);r2=random.Random(77)
   old=DeepSearch(p,3600000);move=old.act(s,r1)
   new=FrontierSearch(p,3600000);got=run_frontier([new.act(s,r2)],FrontierEvaluator(p))[0]
   assert move==got and r1.getstate()==r2.getstate() and before==key(s)
   for k in ('depth','nodes','cutoffs','cache_hits','root_scores'):assert old.stats.get(k)==new.stats.get(k),(k,old.stats,new.stats)
   other=FrontierSearch(p,3600000);gm=run_frontier([other.act(s,random.Random(77))],FrontierEvaluator(gp))[0]
   err=max([abs(a-b) for a,b in zip(old.stats.get('root_scores',[]),other.stats.get('root_scores',[]))] or [0])
   assert err<=2e-5
   results.append(dict(seed=seed,turn=s.turn,done=s.done,cpu_exact=True,gpu_move_equal=gm==move,root_error=err))
 out=dict(probes=results,max_error=max(x['root_error'] for x in results),move_agreement=sum(x['gpu_move_equal'] for x in results)/len(results));assert out['move_agreement']>=.99
 ref.write('artifacts/gpu-frontier/check.json',out);print(out)
if __name__=='__main__':main()
