import random
import pytest
import numpy as np
from blokus.engine import State,legal_moves
from blokus.value_search import BudgetAgent
from blokus.reply_search import ReplyAgent,reset_game
from blokus.aligned_data import BASE
from blokus.node_diagnostic import child,ordered
from blokus.deep_search import key

@pytest.mark.parametrize('width',[3,6,12])
def test_reply_minimax_and_old_parity(monkeypatch,width):
    # Infinite effective time, compare all completed roots to direct two-ply values.
    monkeypatch.setattr('blokus.reply_search.time.perf_counter',lambda:0.)
    s=State(remaining=[7,3]);before=key(s);a=ReplyAgent('value',BASE,.5,24,25,'cpu',reply_width=width)
    moves=ordered(s);vals=[]
    for m in moves:
        c=child(s,m);leaves=[child(c,r) for r in ordered(c)[:width]]
        vals.append(a.evaluator.batch(leaves).min())
    best=np.flatnonzero(np.asarray(vals)>=max(vals)-1e-7)
    expected=moves[int(random.Random(13).choice(best))]
    assert a.act(s,random.Random(13))==expected and a.roots==len(moves) and key(s)==before
    if width==3:assert BudgetAgent('value',BASE,.5,24,25,'cpu').act(s,random.Random(13))==expected

def test_cold_reset_drops_scheduler_history():
    a=ReplyAgent('value',None,budget_ms=25,device='cpu');a.cost_estimate=.01;a.expand_estimate=.02;a.batch_reserve=.03;a.roots=7
    reset_game(a)
    assert (a.cost_estimate,a.expand_estimate,a.batch_reserve)==(.0008,.0005,.0018) and a.roots==7
