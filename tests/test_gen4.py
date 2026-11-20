import copy, random
import numpy as np
import pytest
import torch
from blokus.gen4 import load, search, verify, ROOT
from blokus.value_inputs import encode
from blokus.value_search import encode as old_encode
from blokus.placement_value import PlacementValue, PlacementEvaluator
from blokus.ppo_search_deeper import DeepSearch
from blokus.engine import State, legal_moves, play
from blokus.web_play import Games
from blokus.game_archive import Archive, identity, replay
from test_web_linear import wait
from test_web_play import First

def test_frozen_source_inputs_values_and_complete_tree():
 torch.set_num_threads(1);m=verify();s=State();rng=random.Random(445)
 positions=[]
 for ply in range(39):
  if ply in (0,12,24,36):positions.append(copy.deepcopy(s))
  if s.done:break
  qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None,validate=True)
 for row in m['models']:
  assert (ROOT/row['value']['path']).read_bytes()==(ROOT/row['value']['source']).read_bytes()
  pair=load(row['seed']);p,v=pair
  old=PlacementValue(p.model,'A');old.load_state_dict(torch.load(ROOT/row['value']['source'],weights_only=True)['model']);ref=PlacementEvaluator(old,'native')
  for z in positions:
   for a,b in zip(encode(z),old_encode(z)):np.testing.assert_array_equal(a,b)
   assert v(z)==ref(z)
  # Fixed completed tree equality, not a deadline-sensitive move assertion.
  a=DeepSearch(p,60000,width=4,max_depth=6,value_evaluator=ref)
  b=search(pair);b.budget_ms=60000
  assert a.act(positions[-1],random.Random(0))==b.act(positions[-1],random.Random(0))
  assert a.stats['depth']==b.stats['depth']==6
  assert a.stats['root_scores']==b.stats['root_scores']
  assert a.stats['nodes']==b.stats['nodes']
 with pytest.raises(ValueError,match='seed'):load(1401)

def test_web_gen4_archive_pass_and_budget(tmp_path):
 class Fake(First):
  stats=dict(depth=6,stages=[2,4,6],fallback=False)
 games=Games(First(),archive=Archive(tmp_path),metadata=identity(),record_kind='test',gen4_factory=Fake)
 try:
  with pytest.raises(ValueError):games.create(0,'gen4',2000)
  for seat in (0,1):
   g=games.create(seat,'gen4',500);g=wait(games,g['id'])
   assert g['generation']=='gen4' and g['model_seed']==1751401
   games.move(g['id'],g['revision'],g['legal'][0]);g=wait(games,g['id'])
   assert not g['archive']['failed']
  key=g['id']
  with games.lock:
   games.games[key]['state']=State(remaining=[0,0],last=[0,1]);games.advance(key)
  g=games.snapshot(key)
  assert g['done'] and g['scores']==[20,15]
  assert all(x['forced_pass'] for x in g['move_stats'][-2:])
 finally:games.worker.shutdown()
