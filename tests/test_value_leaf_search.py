import copy,random
import numpy as np
import torch
from blokus.engine import State,play,legal_moves
from blokus.ppo_policy import PPOAgent,collate
from blokus.ppo_value_leaf import value_only,value_inputs
from blokus.ppo_search import SearchAgent
from blokus.ppo_search_staged import ValueLeafSearch,key
from tests.test_deep_search import oracle

def test_value_has_no_candidate_dependency_and_preserves_features(monkeypatch):
 torch.set_num_threads(1)
 for gen,seed in [('gen3',1751401),('gen2',1401)]:
  p=PPOAgent(seed,'cpu',generation=gen);s=State(remaining=[1,0],last=[None,1],turn=1)
  obs,_=p._observation(s)
  with torch.inference_mode():_,v=p.model(*collate([obs],'cpu'))
  b,st=value_inputs(s,p.channels);assert np.array_equal(b,obs[0]) and np.array_equal(st,obs[1])
  monkeypatch.setattr(p,'_observation',lambda *args:(_ for _ in ()).throw(AssertionError('policy observation must not be called')))
  assert value_only(p,s)==v.item()
  s.passes=2;assert value_only(p,s)==np.sign(s.score(s.turn)-s.score(1-s.turn))

def test_completed_widening_fallback_and_cold_cache(monkeypatch):
 import blokus.ppo_search_staged as module
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu');s=State();r=random.Random(99);expected=ValueLeafSearch(p,5000).act(s,random.Random(99));original=module.child;count=0
 def cutoff(z,m):
  nonlocal count
  if key(z)==key(s):
   count+=1
   if count==13:raise TimeoutError
  return original(z,m)
 monkeypatch.setattr(module,'child',cutoff);d=ValueLeafSearch(p,5000,root_stages=(8,12),reuse=True);before=key(s)
 assert d.act(s,r)==expected and d.stats['depth']==4 and d.stats['root_width']==4 and key(s)==before
 assert d.stats['stages']==[dict(depth=2,width=4),dict(depth=4,width=4)]
 monkeypatch.setattr(module,'child',original);z=State(remaining=[1,0],last=[None,1]);m=d.act(z,random.Random(99));assert m==ValueLeafSearch(p,5000).act(z,random.Random(99))

def test_small_all_legal_wdl_oracle_and_bonus():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu')
 for s in [State(remaining=[1,1]),State(remaining=[1,0],last=[None,1]),State(remaining=[1,0],last=[None,1],turn=1),State(remaining=[0,0],last=[0,1])]:
  v,best,n=oracle(s);before=key(s);a=ValueLeafSearch(p,5000,root_stages=(8,12),reuse=True);m=a.act(s,random.Random(3));assert m in best and key(s)==before
  if not a.stats['forced_pass']:assert max(a.stats['root_scores'])==np.sign(v)

def test_auto_backend_fallback(monkeypatch):
 from blokus import ppo_native
 torch.set_num_threads(1);monkeypatch.setattr(ppo_native,'library',lambda:None)
 a=PPOAgent(1751401,'cpu',backend='auto');b=PPOAgent(1751401,'cpu',backend='python');s=State()
 assert value_only(a,s)==value_only(b,s)
 assert ValueLeafSearch(a,5000,width=2,max_depth=2).act(s,random.Random(1))==ValueLeafSearch(b,5000,width=2,max_depth=2).act(s,random.Random(1))
