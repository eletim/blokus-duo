"""New exports, unchanged gen3, CPU/CUDA tolerance fixed at 2e-5 absolute."""
import os,json,random,copy
from pathlib import Path
import numpy as np
import torch
import placement_large_train as t
from blokus.engine import State,play,legal_moves,is_legal
from blokus.value_search import encode
from blokus.placement_value import PlacementValue,PlacementEvaluator,mobility
from blokus.ppo_search_deeper import DeepSearch
R=t.R

def main():
 t.e.setup();rng=random.Random(885600019);states=[]
 for _ in range(3):
  s=State()
  while not s.done:
   if len(states)<120:states.append(copy.deepcopy(s))
   qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None,validate=True)
  states.append(copy.deepcopy(s))
 rows=[]
 for seed in t.e.SEEDS:
  p=t.PPOAgent(seed,'cpu',backend='python');before=copy.deepcopy(p.model.state_dict())
  for variant in 'AC':
   ck=torch.load(R/f'{variant}-{seed}.pt',map_location='cpu',weights_only=True);m=PlacementValue(p.model,variant);initial=copy.deepcopy(m.state_dict());m.load_state_dict(ck['model']);changed=sum(not torch.equal(initial[k],v) for k,v in m.state_dict().items());assert changed>0;cpu=PlacementEvaluator(m,backend='python');gpu=PlacementEvaluator(copy.deepcopy(m).to('cuda'),backend='native');maxerr=0.
   for s in states:
    np.testing.assert_array_equal(mobility(s,'python'),mobility(s,'native'));err=abs(cpu(s)-gpu(s));maxerr=max(maxerr,err);assert err<=2e-5,(seed,variant,err)
   search=DeepSearch(p,30000,max_depth=2,value_evaluator=cpu);s=states[12];move=search.act(s,random.Random(2));assert is_legal(s,move) and search.stats['depth']==2
   assert all(torch.equal(v,before[k]) for k,v in p.model.state_dict().items());rows.append(dict(seed=seed,variant=variant,sha=t.e.sha(R/f'{variant}-{seed}.pt'),source_sha=ck['source_sha'],cpu_cuda_max_abs=maxerr,legal_complete_depth2=True,trained_tensors_changed=changed,updates=ck['updates'],epochs=ck['epochs']))
 t.e.write(R/'export-check.json',dict(pass_=True,states=len(states),cpu_cuda_abs_tolerance=2e-5,models=rows,policy_unchanged=True))
 print('EXPORT CHECK PASS',flush=True)
if __name__=='__main__':main()
