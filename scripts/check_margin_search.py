"""Bounded fixed-depth probe: point units at terminals and komi invariance."""
import copy,json,random
from pathlib import Path
import torch
from blokus.engine import State,Move,play
from blokus.gen4 import load
from blokus.margin_search import MarginSearch
from blokus.margin_value import MarginValue
R=Path('artifacts/margin-pilot');torch.set_num_threads(1)
g=json.loads(next((R/'matches').glob('paired-*.json')).read_text());s=State()
for m in g['moves'][:-8]:play(s,Move(*m) if m is not None else None,validate=True)
p,_=load(1751401,'native');results=[]
for k in (0,4):
 a=MarginSearch(p,60000,4,2,value_evaluator=MarginValue(R/'model.json',k));m=a.act(copy.deepcopy(s),random.Random(17))
 results.append(dict(komi=k,move=None if m is None else [m.piece,m.orientation,m.row,m.col],stats=a.stats))
assert results[0]['move']==results[1]['move']
if 'root_scores' in results[0]['stats']:
 shift=-4 if s.turn==0 else 4
 assert all(abs(b-a-shift)<1e-8 for a,b in zip(results[0]['stats']['root_scores'],results[1]['stats']['root_scores']))
print(json.dumps(dict(turn=s.turn,results=results,passed=True),indent=2))
