import random
import numpy as np
import pytest
import torch
from blokus import ai,root_policy,terminal_ppo as p,cpu_reference as ref
from blokus.engine import State,PLACEMENTS,ORIENTS,legal_moves,is_legal,play,Move
from blokus.placement_geometry import geometry
from blokus.research import clone
from scripts.bench_cpu_throughput import backend,dataset,MODEL
from test_deep_search import independent_child


def same(a,b):
 assert len(a)==len(b)
 for x,y in zip(a,b):assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()

def test_all_placements_orientations_boundaries_exact():
 # Include every board placement, including all orientations and all boundaries.
 assert {q.move.piece for q in PLACEMENTS}==set(range(21))
 for turn in [0,1]:
  s=State(turn=turn)
  same(ai.features(s,PLACEMENTS),ref.features(s,PLACEMENTS))
  same(root_policy.inputs(s,PLACEMENTS),ref.inputs(s,PLACEMENTS))
  actual,_=p.observe(s,PLACEMENTS);expected,_=ref.observe(s,PLACEMENTS);same(actual,expected)
 assert geometry.cache_info().maxsize==32768

def test_independent_games_order_features_transition():
 for s in dataset():
  qs=legal_moves(s);assert qs==ref.legal_moves(s)
  same(ai.features(s,qs),ref.features(s,qs));same(root_policy.inputs(s,qs),ref.inputs(s,qs));same(p.observe(s)[0],ref.observe(s)[0])
  for q in qs:assert is_legal(s,q.move)
  for q in qs[:2]:
   child=clone(s);play(child,q.move);assert child==independent_child(s,q.move)
  if not qs:assert not any(is_legal(s,q.move) for q in PLACEMENTS)

def test_no_shared_mutable_output_or_state_cache():
 s=State();original=clone(s);a,qs=p.observe(s);expected=ref.observe(s)[0]
 for x in a:x.fill(0)
 assert s==original;same(p.observe(s)[0],expected)
 for turn in [0,1]:
  s.turn=turn;s.remaining[turn]=1;s.last[1-turn]=0;s.passes=1
  same(p.observe(s)[0],ref.observe(s)[0])
 s.passes=2
 with pytest.raises(ValueError):p.observe(s)

def test_cold_and_warm_geometry_rng_and_outputs():
 s=State();rng=random.getstate();numpy=np.random.get_state();geometry.cache_clear();cold=p.observe(s)[0];warm=p.observe(s)[0];same(cold,warm)
 assert random.getstate()==rng and np.array_equal(np.random.get_state()[1],numpy[1])

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_network_and_choice_bit_equal(device):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 ai.seed_all(1730000);agent=p.TerminalAgent(MODEL,device)
 for s in dataset()[::8]:
  with backend(True):old=p.observe(s)[0];move=agent.act(s,random.Random(8))
  with backend(False):new=p.observe(s)[0];assert move==agent.act(s,random.Random(8))
  with torch.no_grad():a=agent.model(*p.collate([old],device));b=agent.model(*p.collate([new],device))
  assert all(torch.equal(x,y) for x,y in zip(a,b))
