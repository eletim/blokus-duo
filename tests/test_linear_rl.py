import random
import numpy as np
from blokus.engine import State,legal_moves,play
from blokus.linear_rl import LinearAgent,Trainer,probabilities,inputs

def test_score_matches_initial_heuristic():
 from blokus.ai import heuristic,features
 s=State();rng=random.Random(98)
 for _ in range(30):
  x,qs,phase=inputs(s);assert np.allclose(x@LinearAgent().weights[phase],heuristic(features(s,qs)[0]),atol=3e-6)
  play(s,rng.choice(qs).move if qs else None)

def test_policy_gradient_finite_difference():
 rng=np.random.default_rng(12);x=rng.normal(size=(6,10));w=rng.normal(size=10);p=probabilities(x,w);g=x[2]-p@x
 for i in range(10):
  delta=np.zeros(10);delta[i]=1e-5
  numerical=(np.log(probabilities(x,w+delta)[2])-np.log(probabilities(x,w-delta)[2]))/2e-5
  assert abs(numerical-g[i])<1e-7

def test_resume_identical(tmp_path):
 a=Trainer(31);a.step();a.save(tmp_path/'state.json');b=Trainer.load(tmp_path/'state.json')
 assert a.step()==b.step()
 for k in ('m','v','past'):assert np.array_equal(getattr(a,k),getattr(b,k))
 assert np.array_equal(a.agent.weights,b.agent.weights)
 assert a.rng.bit_generator.state==b.rng.bit_generator.state
 a.agent.save(tmp_path/'policy.json');assert np.array_equal(a.agent.weights,LinearAgent.load(tmp_path/'policy.json').weights)

def test_terminal_pass():
 s=State(remaining=[0,0]);a=LinearAgent();assert a.act(s,random.Random(1)) is None
 move,g=a.sample(s,np.random.default_rng(1));assert move is None and not g.any()

def test_native_matches_python_when_available():
 import pytest
 from blokus.ppo_native import library
 if library() is None:pytest.skip('optional native backend is not built')
 a=Trainer(41,'python');b=Trainer(41,'native')
 for _ in range(2):assert a.step()==b.step()
 assert np.array_equal(a.agent.weights,b.agent.weights)
 assert a.rng.bit_generator.state==b.rng.bit_generator.state
