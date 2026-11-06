"""Export inference-reference fixtures from frozen research implementation."""
import copy,hashlib,json,random,sys
from pathlib import Path
import numpy as np
from blokus.engine import State,play,legal_moves
from blokus.linear_value import Agent
from blokus.linear_tree import CompletedSearch
from scripts import linear_L_confirm as ev
root=Path(sys.argv[1]);states=[];s=State();rng=random.Random(2510000000)
for ply in range(36):
 if ply%6==0:states.append(copy.deepcopy(s))
 if s.done:break
 qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None)
states += [State(remaining=[0,0],last=[0,1]),State(remaining=[0,0],last=[0,1],passes=2)]
rows=[]
for seed in ev.SEEDS:
 for backend in ('python','native'):
  a=Agent.load(ev.source('L',seed),backend)
  for i,s in enumerate(states):
   x,qs,p=a.inputs(s);m=a.act(s,random.Random(11));row=dict(seed=seed,backend=backend,state=[s.boards,s.remaining,s.last,s.turn,s.passes],input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),phase=p,value=a.value(s),move=None if m is None else [m.piece,m.orientation,m.row,m.col])
   if i==3:
    b=CompletedSearch(a,ms=0,width=24,replies=8);m=b.act(s,random.Random(11));row['search']=dict(move=[m.piece,m.orientation,m.row,m.col] if m else None,values=b.stats[-1]['values'])
   rows.append(row)
(root/'models/linear-L/reference.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'tests/test_linear_L_export.py').write_text('''import hashlib,json,random
from pathlib import Path
from blokus.engine import State
from blokus.linear_value import Agent
from blokus.linear_tree import CompletedSearch
ROOT=Path(__file__).resolve().parents[1]
def test_exports_and_source_outputs():
 for d in json.loads((ROOT/'models/linear-L/manifest.json').read_text()):assert hashlib.sha256((ROOT/d['path']).read_bytes()).hexdigest()==d['sha256']
 for r in json.loads((ROOT/'models/linear-L/reference.json').read_text()):
  a=Agent.load(ROOT/f"models/linear-L/seed-{r['seed']}.json",r['backend']);s=State(*r['state']);x,qs,p=a.inputs(s)
  assert hashlib.sha256(x.tobytes()).hexdigest()==r['input_sha256'] and p==r['phase'] and a.value(s)==r['value']
  m=a.act(s,random.Random(11));assert (None if m is None else [m.piece,m.orientation,m.row,m.col])==r['move']
  if 'search' in r:
   b=CompletedSearch(a,ms=0,width=24,replies=8);m=b.act(s,random.Random(11));assert (None if m is None else [m.piece,m.orientation,m.row,m.col])==r['search']['move'] and b.stats[-1]['values']==r['search']['values']
''')
