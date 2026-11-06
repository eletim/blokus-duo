import pytest
from blokus.web_play import Games
from blokus.engine import State
from test_web_play import First
from test_web_linear import wait


def test_deep_route_budget_and_trace(monkeypatch):
 import blokus.web_play as web
 class Fake(First):
  def __init__(self,bot,**kw):
   assert kw==dict(budget_ms=500,width=4,max_depth=6)
   self.stats=dict(depth=4,elapsed_ms=12,stages=[2,4],fallback=False)
 monkeypatch.setattr(web,'DeepSearch',Fake)
 games=Games(First())
 try:
  with pytest.raises(ValueError):games.create(0,'deep',2000)
  g=games.create(1,'deep',500);g=wait(games,g['id'])
  assert g['model_seed']==1751401 and g['generation']=='gen3'
  assert g['move_stats'][0]['depth']==4 and g['move_stats'][0]['elapsed_ms']>=0
  g['move_stats'][0]['depth']=99
  assert games.snapshot(g['id'])['move_stats'][0]['depth']==4
  games.move(g['id'],g['revision'],g['legal'][0]);g=wait(games,g['id'])
  assert g['move_stats'][1] is None
  assert len(g['history'])==len(g['move_stats'])
 finally:games.worker.shutdown()


def test_deep_forced_pass_trace():
 games=Games(First())
 try:
  g=games.create(0,'deep',500)
  with games.lock:
   games.games[g['id']]['state']=State(remaining=[0,0],last=[0,1])
   games.advance(g['id'])
  g=games.snapshot(g['id'])
  assert g['done'] and g['scores']==[20,15]
  assert len(g['move_stats'])==2 and all(x['forced_pass'] for x in g['move_stats'])
 finally:games.worker.shutdown()
