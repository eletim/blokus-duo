import random
import numpy as np
import pytest
from blokus.refinement_search import RefinementAgent,LimitedTree,NodeLimit
from blokus.deep_search import SearchTimeout,key
from blokus.value_search import ValueEvaluator
from blokus.engine import State,legal_moves
from tests.test_deep_search import oracle,reachable_small,independent_child

def test_full_oracle_pass_bonus_and_opponent():
    for s in reachable_small()+[State(remaining=[1,0],last=[None,1],turn=1),State(remaining=[0,0],last=[0,1])]:
        v,optimal,_=oracle(s);t=LimitedTree(ValueEvaluator(None,'cpu'));before=key(s)
        tv,m=t.root(s,[q.move for q in legal_moves(s)] or [None],0,True)
        assert tv==v and m in optimal and key(s)==before and not t.tt

def test_fixed_nodes_reproducible_unknown_excluded():
    s=State(remaining=[3,3]);results=[]
    for _ in range(2):
        a=RefinementAgent(device='cpu',node_limit=100,method='priority')
        m=a.act(s,random.Random(1));results.append((m,a.last_values,a.decisions[-1]['nodes']))
        assert a.decisions[-1]['nodes']<=100
        assert all(i in a.initial_values for i in a.last_values)
    assert results[0]==results[1]

def test_interrupted_refinement_keeps_baseline(monkeypatch):
    s=State(remaining=[1,1]);original=LimitedTree.search
    def stop(self,s,depth,*args,**kwargs):
        if depth==3:raise SearchTimeout()
        return original(self,s,depth,*args,**kwargs)
    monkeypatch.setattr(LimitedTree,'search',stop)
    a=RefinementAgent(device='cpu',node_limit=10000);a.act(s,random.Random(4))
    assert a.last_values==a.initial_values and a.decisions[-1]['attempted']==1
    assert a.decisions[-1]['baseline_retained']

def test_late_values_never_committed():
    now=[0.]
    class Late:
        def batch(self,states):now[0]=2.;return np.ones(len(states))*999
    t=LimitedTree(Late(),clock=lambda:now[0]);t.deadline=1.
    with pytest.raises(SearchTimeout):t.search(State(remaining=[1,1]),1)
    assert not t.tt

def test_no_budget_fallback_pass_and_cold():
    s=State();a=RefinementAgent(device='cpu',node_limit=1)
    m=a.act(s,random.Random(2));assert m in [q.move for q in legal_moves(s)]
    assert not a.last_values and a.decisions[-1]['fallback']
    assert a.act(State(remaining=[0,1]),random.Random(2)) is None
    a.cost_estimate=1.;a.refine_estimate=2.;a.reset_game()
    assert a.cost_estimate==.0008 and a.refine_estimate==.008

def test_initial_values_are_old_three_reply_minimum():
    from blokus.ai import features,heuristic
    from blokus.research import clone
    from blokus.engine import play
    s=State(remaining=[3,3]);a=RefinementAgent(device='cpu',node_limit=10000,method='control')
    qs=legal_moves(s);a.act(s,random.Random(0))
    for i,v in a.initial_values.items():
        c=clone(s);play(c,qs[i].move,validate=False);replies=legal_moves(c)
        ms=[replies[int(j)].move for j in np.argsort(-heuristic(features(c,replies)[0]),kind='stable')[:3]] if replies else [None]
        leaves=[]
        for m in ms:
            leaf=clone(c);play(leaf,m,validate=False);leaves.append(leaf)
        assert abs(v-a.evaluator.batch(leaves).min()*40)<1e-6
    assert len(a.initial_values)==len(qs) and a.decisions[-1]['refined']==0
