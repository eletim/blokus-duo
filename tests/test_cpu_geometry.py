import random
import numpy as np
import pytest
import torch
from blokus import ai,ppo_policy as p,cpu_reference as ref
from blokus.engine import State,PLACEMENTS,legal_moves,is_legal,play
from blokus.research import clone
from blokus.placement_geometry import geometry
from scripts.bench_ppo_cpu import backend,states

def same(a,b):
 for x,y in zip(a,b):assert x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()

def test_all_placements_features_and_pack():
 for turn in [0,1]:
  s=State(turn=turn);same(ai.features(s,PLACEMENTS),ref.features(s,PLACEMENTS));same(p.observe(s,PLACEMENTS)[0],ref.observe(s,PLACEMENTS)[0])

def test_saved_game_order_and_coordinate_checker():
 for s in states():
  qs=legal_moves(s);assert qs==ref.legal_moves(s);same(p.observe(s)[0],ref.observe(s)[0])
  for q in qs:assert is_legal(s,q.move)
  if not qs:assert not any(is_legal(s,q.move) for q in PLACEMENTS)

def test_ownership_and_cache_state_independence():
 s=State();original=clone(s);geometry.cache_clear();a=p.observe(s)[0]
 for x in a:x.fill(0)
 assert s==original;same(p.observe(s)[0],ref.observe(s)[0])
 s.turn=1;s.remaining[1]=1;s.last[0]=0;s.passes=1;same(p.observe(s)[0],ref.observe(s)[0]);assert geometry.cache_info().maxsize==32768

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_network_and_rng_exact(device):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
 ai.seed_all(1730000);agent=p.PPOAgent(1401,device,generation="gen2")
 for s in states()[::8]:
  with backend(True):old=p.observe(s)[0];rng=random.Random(9);m=agent.act(s,rng);r=rng.getstate()
  with backend(False):new=p.observe(s)[0];rng=random.Random(9);assert agent.act(s,rng)==m and rng.getstate()==r
  with torch.no_grad():a=agent.model(*p.collate([old],device));b=agent.model(*p.collate([new],device))
  assert all(torch.equal(x,y) for x,y in zip(a,b))
