import random
import numpy as np
import torch
import pytest
from blokus.ai import seed_all
from blokus.engine import State,legal_moves,play,is_legal
from blokus.research import clone
from blokus.ppo_short_search import ShortSearch,terminal_value,choose_complete
from blokus.terminal_ppo import TerminalAgent,observe,collate
from test_deep_search import oracle,independent_child,independent_moves

PATH='artifacts/ppo-gen2/seed-1401.pt'
@pytest.fixture
def agent():
 seed_all(1);return ShortSearch(PATH,25,device='cpu')

def test_mask_and_same_value_head_and_perspective(agent):
 s=State();o,qs=observe(s)
 with torch.no_grad():logits,v=agent.model(*collate([o],agent.device))
 assert len(logits[0])==len(qs)
 assert agent.values([s],0)[0]==pytest.approx(v.item(),abs=1e-6)
 assert agent.values([s],1)[0]==pytest.approx(-v.item(),abs=1e-6)
 assert abs(v.item())<=1

def test_bonus_and_pass():
 s=State(remaining=[0,0],last=[0,1]);assert terminal_value(s,0)==1 and terminal_value(s,1)==-1
 play(s,None,validate=False);assert not s.done
 play(s,None,validate=False);assert s.done and terminal_value(s,0)==1
 s.last=[0,0];assert terminal_value(s,0)==0

def test_incomplete_and_margin():
 assert choose_complete([-.5,None],0)==0
 assert choose_complete([.1,.2],.15)==0
 assert choose_complete([.1,.3],.15)==1
 assert choose_complete([.1,.1],0)==0

def test_timeout_preserves_identical_greedy_and_state(agent):
 s=State();before=clone(s);agent.budget_ms=0
 other=TerminalAgent(PATH,'cpu')
 assert agent.act(s,random.Random(7))==other.act(s,random.Random(7))
 assert s==before and agent.records[-1]['fallback']
 agent.cost['prepare']=100.;agent.reset_game();assert agent.cost['prepare']<1 and not agent.records

def test_complete_reply_min_and_legal(agent):
 s=State(remaining=[3,3]);agent.budget_ms=1000
 m=agent.act(s,random.Random(1));assert is_legal(s,m)
 assert agent.records[-1]['completed'];assert all(-1<=v<=1 for v in agent.records[-1]['values'])

def test_small_full_endgames_independent_oracle(agent):
 # Deliberately tiny inventories, all placement decisions fit in two plies.
 for rem in ([1,1],[1,2],[2,1],[1,0]):
  for turn in [0,1]:
   s=State(remaining=list(rem),last=[None,1],turn=turn)
   expected,best,_=oracle(s);moves=independent_moves(s) or [None];values=[]
   for m in moves:
    child=independent_child(s,m);replies=independent_moves(child) or [None]
    leaves=[independent_child(child,r) for r in replies]
    # When player with no pieces moves first, forced pass still leaves one real placement.
    assert all(not any(z.remaining) or z.done for z in leaves)
    values.append(min(agent.values(leaves,s.turn)))
   assert max(values)==np.sign(expected)
   assert {m for m,v in zip(moves,values) if v==max(values)}==best

def test_late_complete_discarded(agent,monkeypatch):
 import blokus.ppo_short_search as module
 s=State(remaining=[3,3]);fallback=TerminalAgent(PATH,'cpu').act(s,random.Random(9));agent.budget_ms=1000
 original=agent.values
 def late(states,player):
  v=original(states,player);monkeypatch.setattr(module.time,'perf_counter',lambda:1e12);return v
 monkeypatch.setattr(agent,'values',late)
 assert agent.act(s,random.Random(9))==fallback
 assert agent.records[-1]['reason']=='late_complete_batch_discarded'

def test_opponent_backup_is_minimum_not_average(agent,monkeypatch):
 agent.budget_ms=1000;s=State()
 def values(states,player):
  assert len(states)==24 and player==0
  return [x for i in range(8) for x in ([.8,.7,.9] if i==1 else [.9,-.2,.3])]
 monkeypatch.setattr(agent,'values',values)
 move=agent.act(s,random.Random(1));r=agent.records[-1]
 assert r['values']==[-.2,.7,-.2,-.2,-.2,-.2,-.2,-.2]
 assert r['switched'] and is_legal(s,move)
