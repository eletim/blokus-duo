import copy,random
import numpy as np
import pytest
import torch
from blokus.engine import State,play,legal_moves
from blokus.ppo_policy import PPOAgent,collate
from blokus.ppo_search_deeper import DeepSearch,key
from blokus.placement_value import PlacementValue,PlacementEvaluator,mobility
from blokus.value_search import encode

def positions():
 s=State();rng=random.Random(908711);out=[]
 for i in range(40):
  if i%5==0:out.append(copy.deepcopy(s))
  if s.done:break
  qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None,validate=False)
 return out

def test_placement_counts_native_python_and_remaining():
 for s in positions():
  before=key(s);a=mobility(s,'native').reshape(2,21,6)
  np.testing.assert_array_equal(a.reshape(-1),mobility(s,'python'));assert key(s)==before
  for i,p in enumerate((s.turn,1-s.turn)):
   z=copy.deepcopy(s);z.turn=p;qs=legal_moves(z)
   for piece in range(21):
    count=sum(q.move.piece==piece for q in qs)
    assert a[i,piece,0]==bool(count)
    np.testing.assert_allclose(np.expm1(float(a[i,piece,1])*np.log(4097.)),count,atol=.002)
    assert np.isclose(a[i,piece,2:].sum(),bool(count))
    if not (s.remaining[p]>>piece&1):assert not a[i,piece].any()
  z=copy.deepcopy(s);z.turn=1-s.turn
  np.testing.assert_array_equal(mobility(z).reshape(2,21,6),a[::-1])
 assert not mobility(State(passes=2)).any()

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_zero_adapters_preserve_source_and_policy(device):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
 torch.set_num_threads(1)
 for seed in (1751401,1751402,1751403):
  p=PPOAgent(seed,device,backend='native');original=copy.deepcopy(p.model.state_dict())
  for variant in 'ABC':
   m=PlacementValue(p.model,variant).to(device)
   for s in positions():
    obs,_=p._observation(s)
    with torch.no_grad():_,ref=p.model(*collate([obs],device))
    b,st=encode(s);out=m(torch.tensor(b[None,:4],device=device),torch.tensor(st[None],device=device),torch.tensor(mobility(s)[None],device=device))
    torch.testing.assert_close(out,ref,atol=1e-7,rtol=0)
   opt=torch.optim.Adam(m.parameters(),lr=.001);opt.zero_grad();(out-1).square().mean().backward();opt.step()
   assert all(torch.equal(v,original[k]) for k,v in p.model.state_dict().items())

def test_custom_value_fixed_tree_and_internal_cache():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu',backend='native');s=positions()[2];calls=[]
 def evaluator(z):
  calls.append(key(z));return np.tanh((z.score(z.turn)-z.score(1-z.turn))/20)
 a=DeepSearch(p,30000,width=2,max_depth=4,prune=False,value_evaluator=evaluator)
 b=DeepSearch(p,30000,width=2,max_depth=4,value_evaluator=evaluator)
 assert a.act(s,random.Random(1))==b.act(s,random.Random(1));assert calls
 assert a.stats['depth']==b.stats['depth']==4
 np.testing.assert_allclose(a.stats['root_scores'],b.stats['root_scores'],atol=1e-7,rtol=0)
 # Every nonterminal fixed-depth leaf must use the supplied evaluator,
 # even if reached internally during a previous iterative-deepening stage.
 def walk(z,d):
  if z.done:return np.sign(z.score(z.turn)-z.score(1-z.turn))
  if not d:return evaluator(z)
  obs,qs=p._observation(z)
  with torch.no_grad():l,_=p.model(*collate([obs],'cpu'))
  moves=[qs[i].move for i in l[0].argsort(descending=True).tolist()[:2]] if qs else [None]
  vals=[]
  for move in moves:
   child=copy.deepcopy(z);play(child,move,validate=False);vals.append(-walk(child,d-1))
  return max(vals)
 assert np.isclose(max(a.stats['root_scores']),walk(s,4))

def test_custom_timeout_keeps_completed_depth(monkeypatch):
 import blokus.ppo_search_deeper as mod
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu',backend='native');s=positions()[1];before=key(s)
 evaluator=PlacementEvaluator(PlacementValue(p.model,'A'))
 reference=DeepSearch(p,30000,max_depth=4,value_evaluator=evaluator);expected=reference.act(s,random.Random(7));original=mod.child;roots=0
 def interrupt(z,move):
  nonlocal roots
  if key(z)==before:
   roots+=1
   if roots==9:raise TimeoutError
  return original(z,move)
 monkeypatch.setattr(mod,'child',interrupt)
 search=DeepSearch(p,30000,max_depth=6,value_evaluator=evaluator)
 assert search.act(s,random.Random(7))==expected
 assert search.stats['depth']==4 and search.stats['stages']==[2,4] and not search.stats['fallback']
 assert key(s)==before
