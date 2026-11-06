import copy,json,random
import numpy as np
import pytest
from blokus.linear_value import Agent,value_features
from blokus.linear_value_fit import ValueTrainer
from blokus.linear_rl import prefix
from blokus.engine import Move,play

def test_value_only_update_formula_and_reward_view():
 source=Agent('old',value=np.ones((3,8))*.01);t=ValueTrainer(93,source);initial=t.agent.value_weights.copy();log,games=t.step();phis=[];ys=[]
 for g in games:
  s=prefix(g['prefix_seed']);local=[]
  for row in g['moves']:
   if s.turn==g['seat']:local.append(value_features(s))
   play(s,Move(*row) if row else None)
  delta=s.score(g['seat'])-s.score(1-g['seat']);y=float((delta>0)-(delta<0));assert y==g['reward'];assert len(local)==g['learner_states'];phis+=local;ys += [y]*len(local)
 phi=np.array(phis);pred=np.sum(phi*initial,axis=(1,2));grad=np.mean((np.array(ys)-pred)[:,None,None]*phi,axis=0);grad*=min(1,5/max(np.linalg.norm(grad),1e-12));expected=initial+.01*grad/(np.abs(grad)+1e-8)
 assert np.allclose(expected,t.agent.value_weights,atol=1e-12,rtol=0)
 assert np.array_equal(source.weights,t.agent.weights) and np.array_equal(source.value_weights,t.frozen.value_weights)
 assert np.isfinite(log['mse']) and not np.array_equal(initial,t.agent.value_weights)

def test_fit_resume_and_no_generator_feedback(tmp_path):
 t=ValueTrainer(94,Agent('old',value=np.ones((3,8))*.02),arm='S');t.step();p=tmp_path/'state.json';t.save(p);r=ValueTrainer.load(p);x,g=t.step();y,h=r.step();assert x==y and g==h
 assert np.array_equal(t.agent.value_weights,r.agent.value_weights);assert not np.array_equal(t.agent.value_weights,t.frozen.value_weights)
 d=json.loads(p.read_text());d['lr']=.1;p.write_text(json.dumps(d))
 with pytest.raises(ValueError):ValueTrainer.load(p)
