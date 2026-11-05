import random
import pytest
from blokus.engine import State,Move
from blokus.node_diagnostic import NodeSearch,NodeLimit,ordered,child
from blokus.value_search import ValueEvaluator
from blokus.deep_search import key
from tests.test_deep_search import oracle,reachable_small

def test_fixed_node_repeat_and_hard_limit():
    s=State(remaining=[3,3]);ev=ValueEvaluator(None,'cpu');before=key(s)
    a=NodeSearch(ev,23).roots(s,24,12,4);b=NodeSearch(ev,23).roots(s,24,12,4)
    assert a==b and a['nodes']<=23 and a['interrupted'] and key(s)==before

def test_fixed_node_exact_independent_oracle():
    ev=ValueEvaluator(None,'cpu')
    for s in reachable_small()+[State(remaining=[1,0],last=[None,1],turn=1),State(remaining=[3,0],last=[None,1])]:
        v,optimal,_=oracle(s);t=NodeSearch(ev,20000)
        z=t.roots(s,100000,100000,0,full=True)
        assert not z['interrupted'] and max(r['value'] for r in z['values'])==v
        assert (Move(*z['chosen']) if z['chosen'] else None) in optimal

def test_partial_root_not_published():
    z=NodeSearch(ValueEvaluator(None,'cpu'),1).roots(State(remaining=[3,3]),24,12,4)
    assert not z['values'] and z['interrupted'] and z['chosen'] is not None
