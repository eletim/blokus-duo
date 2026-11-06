import hashlib,json,random
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
