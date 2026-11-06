import random
import time
import pytest
from blokus.engine import State, Move, legal_moves, is_legal, play
from blokus.web_play import Games
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search import SearchAgent

class First:
    def act(self,s,rng):
        qs=legal_moves(s)
        return qs[0].move if qs else None

def test_web_rejects_illegal_and_stale():
    games=Games(First())
    try:
        g=games.create(0,'policy',100)
        with pytest.raises(ValueError):games.move(g['id'],0,[0,0,0,0])
        assert games.snapshot(g['id'])['revision']==0
        games.move(g['id'],0,g['legal'][0])
        with pytest.raises(ValueError):games.move(g['id'],0,g['legal'][0])
    finally:games.worker.shutdown()

def test_full_human_session_and_replay():
    games=Games(First())
    try:
        g=games.create(1,'policy',100)
        end=time.monotonic()+10
        while not g['done']:
            assert time.monotonic()<end
            if not g['busy']:
                assert g['turn']==1 and g['legal']
                games.move(g['id'],g['revision'],g['legal'][0])
            time.sleep(.005)
            g=games.snapshot(g['id'])
        s=State()
        for m in g['history']:play(s,Move(*m) if m else None)
        assert s.done and [s.score(0),s.score(1)]==g['scores']
    finally:games.worker.shutdown()

def test_search_deadline_and_pass():
    import torch
    torch.set_num_threads(1)
    bot=PPOAgent(device='cpu')
    search=SearchAgent(bot,10)
    m=search.act(State(),random.Random(1))
    assert is_legal(State(),m)
    assert search.stats['elapsed_ms']<1000
    s=State(remaining=[0,0])
    assert search.act(s,random.Random(1)) is None

def test_search_finishes_comparison():
    import torch
    torch.set_num_threads(1)
    bot=PPOAgent(device='cpu')
    search=SearchAgent(bot,2000,width=2,max_depth=2)
    m=search.act(State(),random.Random(2))
    assert is_legal(State(),m) and search.stats['depth']==2

def test_new_game_isolation():
    games=Games(First())
    try:
        a=games.create(1,'policy',100)
        b=games.create(0,'policy',100)
        time.sleep(.05)
        assert games.snapshot(b['id'])['revision']==0
        assert games.snapshot(a['id'])['revision']>=1
    finally:games.worker.shutdown()
