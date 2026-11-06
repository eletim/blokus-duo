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

def test_completed_tree_values_match_full_forward_reference():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu');s=State();states=[copy.deepcopy(s)]
 for _ in range(10):play(s,p.act(s,random.Random(44)),validate=False)
 states.append(s)
 def eval_full(z):
  if z.done:return [],float(np.sign(z.score(z.turn)-z.score(1-z.turn)))
  obs,qs=p._observation(z)
  with torch.inference_mode():l,v=p.model(*collate([obs],'cpu'))
  return ([qs[i].move for i in l[0].argsort(descending=True).tolist()] if qs else [None]),v.item()
 def visit(z,d):
  moves,v=eval_full(z)
  if d and not z.done:
   values=[]
   for m in moves[:2]:
    c=copy.deepcopy(z);play(c,m,validate=False);values.append(-visit(c,d-1))
   return max(values)
  return v
 for z in states:
  fallback=p.act(z,random.Random(5));moves,_=eval_full(z);roots=moves[:2]
  if fallback not in roots:roots[-1]=fallback
  ref=[]
  for m in roots:
   c=copy.deepcopy(z);play(c,m,validate=False);ref.append(-visit(c,3))
  e=ValueLeafSearch(p,5000,width=2);e.act(z,random.Random(5))
  np.testing.assert_allclose(e.stats['root_scores'],ref,atol=1e-7,rtol=0)

def test_reuse_matches_uncached_completed_wide_tree():
 torch.set_num_threads(1)
 for seed in (1751401,1751402,1751403):
  p=PPOAgent(seed,'cpu');s=State();before=key(s)
  a=ValueLeafSearch(p,5000,root_stages=(8,12),reuse=False)
  b=ValueLeafSearch(p,5000,root_stages=(8,12),reuse=True)
  assert a.act(s,random.Random(8))==b.act(s,random.Random(8))
  assert a.stats['root_width']==b.stats['root_width']==12
  assert a.stats['stages']==b.stats['stages']
  np.testing.assert_allclose(a.stats['root_scores'],b.stats['root_scores'],atol=1e-7,rtol=0)
  assert key(s)==before
