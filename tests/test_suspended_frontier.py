import copy,random
import numpy as np
import torch
from blokus.engine import State,play
from blokus.ppo_policy import PPOAgent
from blokus.ppo_frontier import FrontierSearch,FrontierEvaluator,run_frontier
from blokus.ppo_search_deeper import DeepSearch,key
from blokus.ppo_value_leaf import value_only

def test_reference_and_interleaved_rng():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu',backend='python');states=[State(),State()]
 expected=[];rngs=[]
 for s in states:
  r=random.Random(99);a=DeepSearch(p,60000,max_depth=2);m=a.act(s,r);expected.append((m,a.stats,r.getstate()))
 searches=[FrontierSearch(p,60000,max_depth=2) for _ in states];rngs=[random.Random(99) for _ in states]
 before=[key(s) for s in states]
 got=run_frontier([a.act(s,r) for a,s,r in zip(searches,states,rngs)],FrontierEvaluator(p))
 for i,m in enumerate(got):
  assert m==expected[i][0] and rngs[i].getstate()==expected[i][2] and key(states[i])==before[i]
  assert np.allclose(searches[i].stats['root_scores'],expected[i][1]['root_scores'],atol=2e-5,rtol=0)

def test_pass_terminal_and_timeout():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu',backend='python')
 s=State();s.remaining=[0,0];s.passes=0
 a=FrontierSearch(p,60000);got=run_frontier([a.act(s,random.Random(0))],FrontierEvaluator(p))[0]
 assert got is None and a.stats['forced_pass']
 # A completed game has no policy action; terminal values are exact signed WDL.
 play(s,None,validate=False);play(s,None,validate=False)
 assert s.done and value_only(p,s)==float(np.sign(s.score(s.turn)-s.score(1-s.turn)))
 a=FrontierSearch(p,.000001);s=State();r=random.Random(7);expected=p.act(s,random.Random(7))
 got=run_frontier([a.act(s,r)],FrontierEvaluator(p))[0]
 assert got==expected and a.stats['fallback'] and a.stats['depth']==0

def test_complete_depth_fallback_and_unpruned_tree():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu',backend='native');s=State();r=random.Random(12)
 for _ in range(16):
  if s.done:break
  play(s,p.act(s,r),validate=True)
 if s.done:return
 a=FrontierSearch(p,60000,max_depth=4,prune=True);b=FrontierSearch(p,60000,max_depth=4,prune=False)
 ma=run_frontier([a.act(s,random.Random(5))],FrontierEvaluator(p))[0]
 mb=run_frontier([b.act(s,random.Random(5))],FrontierEvaluator(p))[0]
 assert ma==mb and a.stats['root_scores']==b.stats['root_scores']
 # Interrupt at the first request after the completed depth4 stage.
 h=FrontierSearch(p,60000,max_depth=6);g=h.act(s,random.Random(5));ev=FrontierEvaluator(p)
 try:
  req=next(g)
  while True:
   # Generator's local stats are owned by act, not shared with other games.
   if g.gi_frame.f_locals['stats']['depth']==4:
    try:g.throw(TimeoutError())
    except StopIteration as end:got=end.value
    break
   req=g.send(ev.batch([req])[0])
 except StopIteration as end:
  # A terminal-only deeper tree may finish without another NN request.
  got=end.value
 assert got==ma and h.stats['depth']>=4

def test_independent_small_terminal_oracle():
 from tests.test_deep_search import oracle
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu',backend='native')
 for s in (State(remaining=[1,1]),State(remaining=[1,0],last=[None,1],turn=1),State(remaining=[1,0],last=[None,1]),State(remaining=[0,0],last=[0,1])):
  value,best,_=oracle(s);before=key(s);a=FrontierSearch(p,60000)
  move=run_frontier([a.act(s,random.Random(3))],FrontierEvaluator(p))[0]
  assert move in best and key(s)==before
  if not a.stats['forced_pass']:assert max(a.stats['root_scores'])==np.sign(value)

def test_custom_value_is_rejected_not_silently_ignored():
 import pytest
 with pytest.raises(ValueError,match='frozen'):
  FrontierSearch(None,value_evaluator=lambda state:0.)
