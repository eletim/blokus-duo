import copy,random
import numpy as np
import torch
from blokus.engine import State,play
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.ppo_search_staged import ValueLeafSearch,key
from tests.test_deep_search import oracle

def test_pruned_fixed_tree_and_completed_4_match():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu');s=State();states=[copy.deepcopy(s)]
 for i in range(18):play(s,p.act(s,random.Random(i)),validate=False)
 states.append(s)
 for z in states:
  for depth in (4,6):
   a=DeepSearch(p,20000,width=2,max_depth=depth,prune=False);b=DeepSearch(p,20000,width=2,max_depth=depth)
   assert a.act(z,random.Random(2))==b.act(z,random.Random(2));assert a.stats['depth']==b.stats['depth']==depth
   np.testing.assert_allclose(a.stats['root_scores'],b.stats['root_scores'],atol=1e-7,rtol=0)
  a=ValueLeafSearch(p,5000);b=DeepSearch(p,20000,max_depth=4)
  assert a.act(z,random.Random(2))==b.act(z,random.Random(2));np.testing.assert_allclose(a.stats['root_scores'],b.stats['root_scores'],atol=1e-7,rtol=0)

def test_timeout_preserves_four_and_state(monkeypatch):
 import blokus.ppo_search_deeper as mod
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu');s=State();before=key(s);ref=DeepSearch(p,20000,max_depth=4);m=ref.act(s,random.Random(3));orig=mod.child;n=0
 def cut(z,a):
  nonlocal n
  if key(z)==before:
   n+=1
   if n==9:raise TimeoutError
  return orig(z,a)
 monkeypatch.setattr(mod,'child',cut);d=DeepSearch(p,20000);assert d.act(s,random.Random(3))==m;assert d.stats['depth']==4 and d.stats['stages']==[2,4];assert key(s)==before

def test_terminal_pass_and_small_oracle():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu')
 for s in (State(remaining=[1,1]),State(remaining=[1,0],last=[None,1],turn=1),State(remaining=[1,0],last=[None,1]),State(remaining=[0,0],last=[0,1])):
  v,best,n=oracle(s);a=DeepSearch(p,20000);before=key(s);assert a.act(s,random.Random(3)) in best;assert key(s)==before
  if not a.stats['forced_pass']:assert max(a.stats['root_scores'])==np.sign(v)

def test_width4_depth6_exact_completed_tree():
 torch.set_num_threads(1);p=PPOAgent(1751402,'cpu');s=State();before=key(s)
 a=DeepSearch(p,30000,width=4,max_depth=6,prune=False);b=DeepSearch(p,30000,width=4,max_depth=6,prune=True)
 assert a.act(s,random.Random(4))==b.act(s,random.Random(4))
 assert a.stats['depth']==b.stats['depth']==6
 np.testing.assert_allclose(a.stats['root_scores'],b.stats['root_scores'],atol=1e-7,rtol=0)
 assert b.stats['cutoffs']>0 and key(s)==before
