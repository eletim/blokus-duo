import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from h_balanced import select,STAGES
import pytest

def rows():
 return [dict(game=g,ply=st+t,turn=t,legal_count=2,stats={'depth':6}) for g in range(6) for st in STAGES for t in (0,1)]
def test_equal_labels_and_actual_turn_not_parity():
 r=rows()
 for x in r:x['turn']=1-x['turn']
 arms,dev,c=select(r,4)
 assert len(arms['A'])==len(arms['B'])==12 and len(dev)==24
 assert all(x['turn']==0 for x in arms['A'])
 assert sum(x['turn']==1 for x in arms['B'])==6
 assert [(x['game'],x['ply']//2) for x in arms['A']]==[(x['game'],x['ply']//2) for x in arms['B']]
def test_pass_singleton_terminal_missing_and_timeout_drop_entire_pair():
 r=rows();r[0]['legal_count']=0;r[2]['legal_count']=1;r[4]['stats']['depth']=4;r=[x for x in r if not(x['game']==1 and x['ply']==13)]
 arms,_,c=select(r,4)
 assert all(x['game']!=0 for x in arms['A'])
 assert not any(x['game']==1 and x['ply'] in (12,13) for x in arms['A'])
 with pytest.raises(AssertionError):select(r,6)
def test_game_split_disjoint():
 train={('train',i) for i in range(128)};dev={('dev',i) for i in range(40)}
 assert not train&dev
