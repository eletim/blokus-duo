import copy,hashlib,json,random
import numpy as np
import pytest
from blokus.linear_value import Agent,Search,outcome
from blokus.engine import State,play,legal_moves

def test_pair_schema_hash_export_and_inputs(tmp_path):
 a=Agent('old',backend='python');b=Agent('new',backend='python');b.value_weights[:]=np.arange(24).reshape(3,8)/100
 p=tmp_path/'policy.json';v=tmp_path/'value.json';a.save(p);b.save(v)
 h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 c=Agent.load_pair(p,v,'python',h(p),h(v));s=State()
 assert np.array_equal(c.inputs(s)[0],a.inputs(s)[0]);assert c.value(s)==b.value(s)
 assert np.array_equal(c.weights,a.weights);assert np.array_equal(c.value_weights,b.value_weights)
 q=tmp_path/'hybrid.json';c.save(q,c.provenance);d=Agent.load(q,'python');assert d.value(s)==c.value(s);assert d.act(s,random.Random(1))==c.act(s,random.Random(1))
 assert json.loads(q.read_text())['metadata']==c.provenance
 with pytest.raises(ValueError,match='SHA256'):Agent.load_pair(p,v,'python','0'*64,h(v))
 x=json.loads(v.read_text());x['format']='unknown';v.write_text(json.dumps(x))
 with pytest.raises(ValueError):Agent.load_pair(p,v,'python')

def test_hybrid_independent_small_end_and_deadline():
 # Exhaustive independent recursion, each pass changes player; no learned leaf remains.
 rng=random.Random(77);s=State();a=Agent('old',backend='native');b=Agent('new',value=np.ones((3,8))*.1)
 while not s.done:
  moves=legal_moves(s)
  if sum(x.bit_count() for x in s.remaining)<=3:break
  play(s,rng.choice(moves).move if moves else None)
 def solve(t):
  if t.done:return outcome(t,t.turn)
  ans=[]
  for m in ([q.move for q in legal_moves(t)] or [None]):
   u=copy.deepcopy(t);play(u,m);ans.append(-solve(u))
  return max(ans)
 a.value_weights=b.value_weights.copy();search=Search(a,ms=0,width=None,replies=None,depth=10)
 # Guarantee a nonterminal exhaustive case even if the random game stopped earlier.
 s=State(remaining=[1,1])
 if not s.done:
  m=search.act(s,random.Random(1));t=copy.deepcopy(s);play(t,m);assert -solve(t)==solve(s)
 opening=State();fallback=Search(a,ms=1e-8);before=copy.deepcopy(opening);m=fallback.act(opening,random.Random(3));assert m==a.act(opening,random.Random(3));assert fallback.stats[-1]['fallback'];assert opening==before


def test_hybrid_pass_terminal_and_view(tmp_path):
 from blokus.linear_value import value_features
 a=Agent('old');b=Agent('new',value=np.arange(24).reshape(3,8)/100)
 p=tmp_path/'p.json';v=tmp_path/'v.json';a.save(p);b.save(v);mixed=Agent.load_pair(p,v)
 s=State();play(s,legal_moves(s)[0].move);t=copy.deepcopy(s);t.turn=1-t.turn
 assert mixed.value(s)==b.value(s) and np.array_equal(value_features(s),-value_features(t))
 assert abs(mixed.value(s)+mixed.value(t))<1e-12
 s=State(remaining=[0,1]);assert Search(mixed).act(s,random.Random(1)) is None;play(s,None)
 assert s.turn==1 and not s.done
 s=State(remaining=[0,0],last=[0,1]);play(s,None);play(s,None)
 assert s.score(0)==20 and s.score(1)==15 and mixed.value(s)==outcome(s,s.turn)
