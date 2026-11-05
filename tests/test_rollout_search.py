import random
import pytest
from blokus.engine import State,Move,play,legal_moves
from blokus.rollout_search import candidates,terminal_rollout,allocate,RolloutAgent,RolloutTimeout
from blokus.deep_search import key

def test_terminal_score_perspective_pass_bonus_and_label_replay():
    for turn,expected in [(0,5),(1,-5)]:
        s=State(remaining=[1,0],last=[None,1],turn=turn);m=candidates(s)[0];before=key(s)
        z=terminal_rollout(s,m,19,trace=True)
        assert z['value']==expected and key(s)==before and z==terminal_rollout(s,m,19,trace=True)
        for row in z['moves']:
            m=Move(*row) if row else None
            if m is None:assert not legal_moves(s)
            play(s,m,validate=m is not None)
        assert s.done and [s.score(0),s.score(1)]==z['scores']

def test_late_rollout_never_returns_score():
    ticks=[0]
    def clock():ticks[0]+=1;return ticks[0]
    with pytest.raises(RolloutTimeout):terminal_rollout(State(),candidates(State())[0],0,deadline=4,clock=clock)

@pytest.mark.parametrize('method',['uniform','halving'])
def test_allocation_repeat_crn_and_budget(method):
    s=State(remaining=[7,7]);ms=candidates(s);before=key(s)
    a=allocate(s,ms,3,32,method,trace=True);b=allocate(s,ms,3,32,method,trace=True)
    assert a==b and a['completed']<=32 and not a['fallback'] and key(s)==before
    first=[z['seed'] for z in a['labels'] if z['sample']==0];assert len(set(first))==1 and len(first)==len(ms)

def test_partial_first_round_fallback(monkeypatch):
    calls=[0]
    def fake(*args,**kwargs):
        calls[0]+=1
        if calls[0]>1:raise RolloutTimeout()
        return dict(value=1000,plies=1,moves=None,scores=[0,0])
    monkeypatch.setattr('blokus.rollout_search.terminal_rollout',fake)
    z=allocate(State(),candidates(State()),total=32)
    assert z['fallback'] and z['candidate']==0 and z['completed']==1 and z['committed']==0 and z['discarded_completed']==1

def test_partial_later_round_keeps_completed_comparison(monkeypatch):
    calls=[0]
    def fake(*args,**kwargs):
        i=calls[0];calls[0]+=1
        if i==9:raise RolloutTimeout()
        return dict(value=i if i<8 else 1000,plies=1,moves=None,scores=[0,0])
    monkeypatch.setattr('blokus.rollout_search.terminal_rollout',fake)
    z=allocate(State(),candidates(State()),total=32)
    assert not z['fallback'] and z['candidate']==7 and z['committed']==8 and z['completed']==9 and len(z['values'][0])==1

def test_halving_counts_and_cold_reset(monkeypatch):
    def fake(s,m,seed,*args,**kwargs):return dict(value=m.row*100+m.col,plies=1,moves=None,scores=[0,0])
    monkeypatch.setattr('blokus.rollout_search.terminal_rollout',fake)
    s=State();ms=candidates(s);z=allocate(s,ms,7,32,'halving')
    assert sorted(map(len,z['values']))==[2,2,2,2,4,4,8,8] and z['completed']==32
    a=RolloutAgent(100);a.turn=20;a.reset_game(7);assert a.turn==0 and a.seed==7
    before=allocate(s,ms,a.seed,32,'halving');a.turn=9;a.reset_game(7)
    assert allocate(s,ms,a.seed,32,'halving')==before

def test_label_audit_does_not_mutate_saved_snapshot():
    import copy
    from scripts.audit_rollouts import replay_snapshot
    from blokus.visited_data import state_dict
    s=State(remaining=[1,0],last=[None,1]);snapshot=state_dict(s);before=copy.deepcopy(snapshot)
    z=terminal_rollout(s,candidates(s)[0],7,trace=True)
    a=replay_snapshot(snapshot,z['moves']);b=replay_snapshot(snapshot,z['moves'])
    assert snapshot==before and a.done and key(a)==key(b)
