import random
import time
from functools import lru_cache
import numpy as np
import pytest
from blokus.engine import State,PLACEMENTS,SIZES,legal_moves,is_legal,play,mask
from blokus.research import clone
from blokus.value_search import ValueEvaluator
from blokus.deep_search import TreeSearch,DeepAgent,Entry,EXACT,LOWER,UPPER,SearchTimeout,key
from blokus.aligned_data import BASE


def independent_moves(s):
    occupied=s.boards[0]|s.boards[1]
    return [q.move for q in PLACEMENTS if s.remaining[s.turn]>>q.move.piece&1 and not q.bits&occupied and is_legal(s,q.move)]


def independent_child(s,m):
    z=clone(s);p=z.turn
    if m is None:z.passes+=1
    else:z.boards[p]|=mask(m.cells);z.remaining[p]&=~(1<<m.piece);z.last[p]=m.piece;z.passes=0
    z.turn=1-p;return z


def independent_score(s):
    def one(p):return (20 if s.last[p]==0 else 15) if not s.remaining[p] else -sum(SIZES[i] for i in range(21) if s.remaining[p]>>i&1)
    return one(s.turn)-one(1-s.turn)


def oracle(s):
    memo={};nodes=0
    def visit(z):
        nonlocal nodes
        k=(tuple(z.boards),tuple(z.remaining),tuple(z.last),z.turn,z.passes)
        if k in memo:return memo[k]
        nodes+=1
        if nodes>20000:raise RuntimeError('oracle bound')
        if z.passes==2:v=independent_score(z)
        else:v=max(-visit(independent_child(z,m)) for m in independent_moves(z) or [None])
        memo[k]=v;return v
    moves=independent_moves(s) or [None];values=[-visit(independent_child(s,m)) for m in moves];best=max(values)
    return best,{m for m,v in zip(moves,values) if v==best},nodes


def reachable_small():
    states=[]
    for seed in [881,882,883]:
        rng=random.Random(seed);s=State();history=[]
        while not s.done:
            qs=legal_moves(s);history.append(clone(s));play(s,rng.choice(qs).move if qs else None,validate=False)
        states.extend(history[-3:])
    return states


def test_exact_matches_independent_all_moves_and_bonus():
    # Reachable late states plus deliberately tiny bonus/pass scenarios.
    states=reachable_small()+[State(remaining=[0,0],last=[0,1]),State(remaining=[1,0],last=[None,1]),State(remaining=[1,2],last=[None,None]),State(remaining=[3,0],last=[None,1])]
    t=TreeSearch(ValueEvaluator(None,'cpu'));solved=0
    for s in states:
        before=key(s);v,best,n=oracle(s);moves=[q.move for q in legal_moves(s)] or [None]
        tv,move=t.root(s,moves,0,True)
        assert tv==v and move in best and key(s)==before
        # Independently compare the complete optimal move set using full windows.
        values={m:-t.search(independent_child(s,m),0,full=True) for m in moves}
        assert {m for m,x in values.items() if x==max(values.values())}==best
        solved+=1
    assert solved==13


def test_tt_bound_and_pass_sign():
    s=State(remaining=[1,0],last=[None,1],turn=1)
    expected,_,_=oracle(s);t=TreeSearch(ValueEvaluator(None,'cpu'));v=t.search(s,0,full=True)
    assert v==expected==-5 # opponent places last mono for20, this side already15
    for a,b,bound in [(expected-3,expected-1,LOWER),(expected+1,expected+3,UPPER),(-100,100,EXACT)]:
        t.tt.clear();t.search(s,0,a,b,True);entry=t.tt[(key(s),-1)]
        assert entry.bound==bound
        assert t.search(s,0,-100,100,True)==expected
    changed=clone(s);changed.last[1]=0;assert key(changed)!=key(s)


def test_timeout_does_not_publish_partial_root_or_entry():
    s=State(remaining=[1,1]);t=TreeSearch(ValueEvaluator(None,'cpu'));t.deadline=0
    with pytest.raises(SearchTimeout):t.search(s,4)
    assert not t.tt


def test_agent_completed_iteration_fallback(monkeypatch):
    s=State(remaining=[1,1]);agent=DeepAgent(BASE,100,deep_gate=42,exact_pieces=0,device='cpu');qs=legal_moves(s);chosen=qs[-1].move;before=key(s)
    def root(state,moves,depth,full=False,preferred=None):
        if depth==2:return 7.,chosen
        raise SearchTimeout()
    monkeypatch.setattr(agent.tree,'root',root)
    assert agent.act(s,random.Random(0))==chosen and key(s)==before
    assert agent.decisions[-1]['completed_depth']==2 and agent.decisions[-1]['value']==7 and not agent.decisions[-1]['exact']


def test_selective_tt_changes_order_not_membership():
    s=State();t=TreeSearch(ValueEvaluator(None,'cpu'),width=2);qs=legal_moves(s);allowed=t.ordered(s,qs)
    outside=next(q.move for q in qs if q.move not in allowed);t.tt[(key(s),1)]=Entry(-1e9,LOWER,outside)
    v=t.search(s,1);ref=TreeSearch(ValueEvaluator(None,'cpu'),width=2).search(s,1)
    assert abs(v-ref)<1e-6


def test_late_leaf_batch_is_discarded():
    clock=[0.]
    class Late:
        def batch(self,states):clock[0]=2.;return np.zeros(len(states))
    tree=TreeSearch(Late(),clock=lambda:clock[0]);tree.deadline=1.
    with pytest.raises(SearchTimeout):tree.search(State(remaining=[3,1]),1)
    assert not tree.tt


def test_hybrid_fallback_matches_complete_old_two_ply(monkeypatch):
    from blokus.value_search import BudgetAgent
    s=State(remaining=[3,1]);before=key(s)
    agent=DeepAgent(BASE,100,exact_pieces=0,exact_mobility=10,only_exact=True,max_depth=2,device='cpu')
    def interrupt_full(*args,**kwargs):raise SearchTimeout()
    monkeypatch.setattr(agent.tree,'root',interrupt_full)
    new=agent.act(s,random.Random(119));old=BudgetAgent('value',BASE,.5,24,100,'cpu');previous=old.act(s,random.Random(119))
    assert old.roots==len(legal_moves(s)) and new==previous and key(s)==before
    assert agent.decisions[-1]['exact_attempt'] and not agent.decisions[-1]['exact'] and agent.decisions[-1]['completed_depth']==2


def test_mobility_gate_certifies_all_legal_tree():
    s=State(remaining=[3,1]);agent=DeepAgent(BASE,100,exact_pieces=0,exact_mobility=10,only_exact=True,max_depth=2,device='cpu')
    m=agent.act(s,random.Random(1));v,optimal,_=oracle(s)
    assert agent.decisions[-1]['exact'] and agent.decisions[-1]['value']==v and m in optimal


def test_loose_tt_lower_bound_is_not_exact_value():
    s=State(remaining=[3,0],last=[None,1]);t=TreeSearch(ValueEvaluator(None,'cpu'))
    mono=next(q.move for q in legal_moves(s) if q.move.piece==0)
    # Force the inferior mono-first line first. It reaches0, but domino-first yields5.
    t.tt[(key(s),-1)]=Entry(-100,LOWER,mono)
    lower=t.search(s,0,-2,-1,True)
    assert lower==0 and t.tt[(key(s),-1)].bound==LOWER
    exact=t.search(s,0,full=True);assert exact==oracle(s)[0]==5
    assert t.tt[(key(s),-1)].bound==EXACT
