import copy,random,threading,time
import numpy as np
import pytest
from blokus.engine import State,Move,play,legal_moves
from blokus.linear_inference import load_search,make_search
from blokus.web_play import Games
from test_web_play import First

def wait(games,key):
 end=time.monotonic()+15
 while True:
  g=games.snapshot(key)
  assert not g['error'],g['error']
  if not g['busy']:return g
  assert time.monotonic()<end
  time.sleep(.005)

@pytest.mark.parametrize('seat',[0,1])
def test_linear_full_session_seats_metadata_and_replay(seat):
 model=load_search(backend='native').agent;games=Games(First(),lambda:make_search(model))
 try:
  g=games.create(seat,'linear',2000)
  assert g['budget']==100 and g['generation']=='linear-L' and g['model_seed']==3101
  while not g['done']:
   g=wait(games,g['id'])
   if g['done']:break
   assert g['turn']==seat and g['legal']
   g=games.move(g['id'],g['revision'],g['legal'][0])
  s=State()
  for m in g['history']:play(s,Move(*m) if m else None)
  assert s.done and g['scores']==[s.score(0),s.score(1)]
  assert 'elapsed_ms' in g['stats'] and 'target_complete' in g['stats']
 finally:games.worker.shutdown()

def test_linear_pass_bonus_session_and_other_session_untouched():
 games=Games(First(),lambda:load_search(backend='python'))
 try:
  g=games.create(0,'linear',100);other=games.create(0,'policy',500)
  with games.lock:
   games.games[g['id']]['state']=State(remaining=[0,0],last=[0,1])
   games.advance(g['id'])
  end=games.snapshot(g['id']);assert end['done'] and end['scores']==[20,15] and end['history']==[None,None]
  assert games.snapshot(other['id'])['revision']==0 and games.snapshot(other['id'])['generation']=='gen3'
 finally:games.worker.shutdown()

def test_replaced_session_discards_inflight_result():
 entered=threading.Event();release=threading.Event()
 class Slow(First):
  def act(self,s,rng):entered.set();assert release.wait(5);return super().act(s,rng)
 games=Games(Slow())
 try:
  a=games.create(1,'policy',100);assert entered.wait(5)
  replacement=copy.deepcopy(games.games[a['id']]);replacement['state']=State();replacement['seat']=0;replacement['busy']=False
  with games.lock:games.games[a['id']]=replacement
  release.set();games.worker.shutdown()
  assert games.snapshot(a['id'])['revision']==0 and games.snapshot(a['id'])['boards']==[[],[]]
 finally:release.set();games.worker.shutdown()

def test_shared_fixed_search_and_factory_validation():
 cli=load_search(3101,'python');web=make_search(cli.agent)
 assert (web.ms,web.width,web.replies,web.depth,web.base_width,web.base_replies)==(100,24,8,2,6,3)
 assert np.array_equal(cli.agent.value_weights,web.agent.value_weights)
 cli.ms=web.ms=0;s=State(remaining=[1,1])
 assert cli.act(s,random.Random(1))==web.act(s,random.Random(1))
 assert cli.stats[-1]['values']==web.stats[-1]['values']
 with pytest.raises(ValueError):load_search(1401)
 games=Games(First())
 try:
  with pytest.raises(ValueError):games.create(0,'linear',100)
 finally:games.worker.shutdown()
