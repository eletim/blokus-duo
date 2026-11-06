import copy,random
import numpy as np
from blokus.engine import State,play,legal_moves,is_legal
from blokus.linear_rl import Trainer as OldTrainer
from blokus.linear_train2 import Trainer
from blokus.linear_value import Agent,Search,enriched_inputs,value_features,outcome,component_sizes

def test_old_control_matches_original():
 a=OldTrainer(41,'native');b=Trainer(41,'old',series=970000000)
 for _ in range(2):
  x=a.step();y,g=b.step();assert x['gradient_norm']==y['gradient_norm'];assert np.array_equal(a.agent.weights,b.agent.weights);assert a.rng.bit_generator.state==b.rng.bit_generator.state

def test_features_native_python_and_initial():
 s=State();a=Agent();rng=random.Random(91)
 for _ in range(30):
  if s.done:break
  x,qs,p=enriched_inputs(s,'native');y,qq,k=enriched_inputs(s,'python');assert qs==qq and p==k and np.array_equal(x,y);assert np.isfinite(x).all()
  old=Agent('old');z,_,_=old.inputs(s);assert np.allclose(x@a.weights[p],z@old.weights[p],atol=1e-12)
  if qs:assert is_legal(s,qs[0].move)
  play(s,rng.choice(qs).move if qs else None)

def test_value_sign_bonus_pass():
 s=State();rng=random.Random(42)
 for _ in range(28):
  if s.done:break
  phi=value_features(s);t=copy.deepcopy(s);t.turn=1-s.turn;assert np.array_equal(phi,-value_features(t))
  qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None)
 s=State(remaining=[0,0],last=[0,1]);play(s,None);assert not s.done;play(s,None);assert outcome(s,0)==1 and outcome(s,1)==-1
 a=Agent(value=np.ones((3,8)));assert a.value(s)==outcome(s,s.turn)

def exact(s):
 if s.done:return outcome(s,s.turn),{None}
 qs=legal_moves(s);moves=[q.move for q in qs] or [None];values=[]
 for m in moves:
  c=copy.deepcopy(s);play(c,m);v,_=exact(c);values.append(-v)
 best=max(values);return best,{m for m,v in zip(moves,values) if v==best}

def test_search_independent_small_end_and_fallback():
 # Only a monomino remains for each side: exhaustive four-ply terminal solution.
 s=State(remaining=[1,1]);before=copy.deepcopy(s);v,moves=exact(s);a=Agent();search=Search(a,ms=0,width=None,replies=None,depth=4)
 m=search.act(s,random.Random(4));assert m in moves and max(search.stats[-1]['values'])==v and s==before
 tiny=Search(a,ms=.000001);assert tiny.act(s,random.Random(5))==a.act(s,random.Random(5));assert tiny.stats[-1]['fallback'] and not tiny.stats[-1]['values']
 s.remaining=[0,1];assert Search(a).act(s,random.Random(0)) is None

def test_resume_value_and_policy(tmp_path):
 a=Trainer(52);a.step();a.save(tmp_path/'state.json');b=Trainer.load(tmp_path/'state.json');x,g=a.step();y,h=b.step();assert x==y and g==h
 for k in ('m','v','vm','vv','past'):assert np.array_equal(getattr(a,k),getattr(b,k))
 assert np.array_equal(a.agent.value_weights,b.agent.value_weights) and np.array_equal(a.agent.weights,b.agent.weights)

def test_search_real_game_small_endings():
 s=State();agent=Agent('old');rng=random.Random(551);states=[]
 while not s.done:
  states.append(copy.deepcopy(s));play(s,agent.act(s,rng))
 checked=0
 for state in states[-4:]:
  if not legal_moves(state):continue
  value,moves=exact(state);search=Search(agent,ms=0,width=None,replies=None,depth=10)
  m=search.act(state,random.Random(0));assert m in moves and max(search.stats[-1]['values'])==value;checked+=1
 assert checked>0

def test_new_feature_policy_gradient_and_value_gradient():
 from blokus.linear_rl import probabilities
 x,qs,p=enriched_inputs(State());w=Agent().weights[p].copy();prob=probabilities(x,w);g=x[3]-prob@x
 for i in range(len(w)):
  delta=np.zeros_like(w);delta[i]=1e-5
  finite=(np.log(probabilities(x,w+delta)[3])-np.log(probabilities(x,w-delta)[3]))/2e-5
  assert abs(finite-g[i])<1e-7
 s=State();play(s,qs[3].move);phi=value_features(s);v=np.full((3,8),.1);target=1.;grad=(target-np.sum(v*phi))*phi
 for i in range(24):
  d=np.zeros((3,8));d.flat[i]=1e-5
  objective=lambda z:-.5*(target-np.sum(z*phi))**2
  assert abs((objective(v+d)-objective(v-d))/2e-5-grad.flat[i])<1e-8

def test_export_and_configuration_validation(tmp_path):
 import pytest
 a=Agent(value=np.ones((3,8)));a.save(tmp_path/'model.json');b=Agent.load(tmp_path/'model.json')
 assert np.array_equal(a.weights,b.weights) and np.array_equal(a.value_weights,b.value_weights)
 with pytest.raises(ValueError):Agent(backend='unknown')
 with pytest.raises(ValueError):Search(a,ms=-1)
 with pytest.raises(ValueError):Search(a,width=0)
