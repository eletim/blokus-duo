import copy,random
import numpy as np
import pytest
from blokus.engine import State,play,legal_moves
from blokus.linear_value import Agent,Search,outcome
from blokus.linear_tree import CompletedSearch

def exact(s):
 # Independent coordinate-set exhaustive solver for monomino-only endings.
 # These tests deliberately do not use engine legal_moves, play, or score here.
 from blokus.engine import Move,cells_of
 assert all(b in (0,1) for b in s.remaining)
 def solve(boards,left,last,turn,passes):
  if passes>=2:
   scores=[(20 if last[p]==0 else 15) if left[p]==0 else -1 for p in (0,1)]
   d=scores[turn]-scores[1-turn];return float((d>0)-(d<0)),{None}
  own,opp=boards[turn],boards[1-turn];moves=[]
  if left[turn]:
   for r in range(14):
    for c in range(14):
     if (r,c) in own|opp:continue
     if any((r+dr,c+dc) in own for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))):continue
     if (not own and (r,c)==((4,4),(9,9))[turn]) or (own and any((r+dr,c+dc) in own for dr,dc in ((1,1),(1,-1),(-1,1),(-1,-1)))):moves.append(Move(0,0,r,c))
  if not moves:moves=[None]
  values=[]
  for m in moves:
   b=[set(x) for x in boards];l=left.copy();la=last.copy();pa=passes+1
   if m is not None:b[turn].add((m.row,m.col));l[turn]=0;la[turn]=0;pa=0
   values.append(-solve(b,l,la,1-turn,pa)[0])
  return max(values),{m for m,v in zip(moves,values) if v==max(values)}
 return solve([set(cells_of(x)) for x in s.boards],s.remaining.copy(),s.last.copy(),s.turn,s.passes)

def test_exact_terminal_pass_and_same_player_continuation():
 a=Agent('old')
 for s in [State(remaining=[1,1]),State(remaining=[1,0],last=[None,1]),State(remaining=[0,0],last=[0,1])]:
  before=copy.deepcopy(s)
  if not legal_moves(s):
   b=CompletedSearch(a,ms=0,width=None,replies=None,depth=4,base_width=None,base_replies=None);assert b.act(s,random.Random(1)) is None
   t=copy.deepcopy(s);play(t,None);assert t.turn!=s.turn
   if not t.done:play(t,None) if not legal_moves(t) else None
  else:
   v,best=exact(s);b=CompletedSearch(a,ms=0,width=None,replies=None,depth=4,base_width=None,base_replies=None);m=b.act(s,random.Random(1));assert m in best and max(b.stats[-1]['values'])==v
  assert s==before
 s=State(remaining=[0,0],last=[0,1],passes=2);assert a.value(s)==1; s.turn=1;assert a.value(s)==-1

def test_completed_stage_fallback_and_no_state_mutation():
 s=State();a=Agent('old');before=copy.deepcopy(s)
 base=CompletedSearch(a,ms=0);m=base.act(s,random.Random(4));n=base.stats[-1]['nodes']
 deeper=CompletedSearch(a,ms=0,depth=4,node_limit=n);assert deeper.act(s,random.Random(4))==m
 stat=deeper.stats[-1];assert stat['depth']==2 and not stat['target_complete'] and stat['values']==base.stats[-1]['values']
 tiny=CompletedSearch(a,ms=.000001);assert tiny.act(s,random.Random(4))==a.act(s,random.Random(4));assert tiny.stats[-1]['fallback'] and tiny.stats[-1]['values']==[]
 assert s==before

def test_fixed_tree_matches_legacy_without_pruning():
 s=State();a=Agent('old');rng=random.Random(123)
 for _ in range(20):play(s,a.act(s,rng))
 for w,r in [(6,3),(24,8)]:
  old=Search(a,ms=0,width=w,replies=r);new=CompletedSearch(a,ms=0,width=w,replies=r,base_width=w,base_replies=r)
  old.act(s,random.Random(0));new.act(s,random.Random(0));assert old.stats[-1]['values']==new.stats[-1]['values']
  stat=new.stats[-1];assert stat['baseline'] in stat['roots'] and len({tuple(x) for x in stat['roots']})==len(stat['roots'])

def test_greedy_near_tie_not_dropped():
 a=Agent('old');a.weights[:]=0;s=State();b=CompletedSearch(a,ms=0,width=1,replies=1,base_width=1,base_replies=1);b.act(s,random.Random(89));assert b.stats[-1]['baseline'] in b.stats[-1]['roots']
 with pytest.raises(ValueError):CompletedSearch(a,depth=3)

def test_wide_stage_partial_results_are_discarded():
 a=Agent('old');s=State();base=CompletedSearch(a,ms=0);m=base.act(s,random.Random(7))
 wide=CompletedSearch(a,ms=0,width=24,replies=8,node_limit=base.stats[-1]['nodes']+10)
 assert wide.act(s,random.Random(7))==m
 assert wide.stats[-1]['values']==base.stats[-1]['values'] and wide.stats[-1]['root_width']==6 and not wide.stats[-1]['target_complete']

def test_depth_four_matches_separate_legacy_fixed_tree():
 s=State();a=Agent('old');rng=random.Random(409)
 for _ in range(20):play(s,a.act(s,rng))
 old=Search(a,ms=0,width=2,replies=2,depth=4)
 new=CompletedSearch(a,ms=0,width=2,replies=2,depth=4,base_width=2,base_replies=2)
 old.act(s,random.Random(2));new.act(s,random.Random(2))
 assert new.stats[-1]['depth']==4 and new.stats[-1]['values']==old.stats[-1]['values']
