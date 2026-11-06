import copy,random
import numpy as np
import torch
from blokus.engine import State,legal_moves,play,is_legal
from blokus.ppo_policy import PPOAgent,collate
from blokus.ppo_search import SearchAgent
from blokus.ppo_search_batch import BatchSearch,key,terminal
from tests.test_deep_search import oracle,reachable_small

def test_batch_outputs_and_complete_tree():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu');s=State();r=random.Random(94);states=[]
 for _ in range(4):
  states.append(copy.deepcopy(s));play(s,p.act(s,r),validate=False)
 obs=[p._observation(s)[0] for s in states]
 with torch.inference_mode():
  l,v=p.model(*collate(obs,'cpu'))
  for i,o in enumerate(obs):
   a,b=p.model(*collate([o],'cpu'));np.testing.assert_allclose(l[i,:len(o[2])],a[0],atol=1e-5,rtol=0);np.testing.assert_allclose(v[i],b[0],atol=2e-6,rtol=0)
 for s in states:
  old=SearchAgent(p,5000,width=2,max_depth=2);new=BatchSearch(p,5000,root_width=2,width=2,max_depth=2);before=key(s)
  assert old.act(s,r)==new.act(s,r);assert new.stats['depth']==2 and key(s)==before

def test_exact_wdl_pass_bonus_oracle():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu')
 states=reachable_small()+[State(remaining=[0,0],last=[0,1]),State(remaining=[1,0],last=[None,1]),State(remaining=[1,0],last=[None,1],turn=1)]
 for s in states:
  v,best,_=oracle(s);a=BatchSearch(p,5000,exact=True);before=key(s);m=a.act(s,random.Random(1))
  # WDL optimum may tie score optima: verify chosen action reaches optimal WDL.
  if sum(x.bit_count() for x in s.remaining)<=6:
   assert a.stats['exact_solved'];c=copy.deepcopy(s);play(c,m,validate=False)
   cv=terminal(c) if c.done else np.sign(oracle(c)[0]);assert -cv==np.sign(v)
  assert key(s)==before and is_legal(s,m)

def test_timeout_fallback_and_cache_cold():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu');s=State();a=BatchSearch(p,.001);r=random.Random(3)
 assert a.act(s,r)==p.act(s,r);assert a.stats['fallback'] and a.stats['depth']==0
 assert a.act(s,r)==p.act(s,r);assert a.stats['nn_calls']==0
