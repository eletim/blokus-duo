import json
import numpy as np
from blokus.cem import INITIAL, ACTIVE, population, update_distribution, linear_checkpoint, train
from blokus.research import Agent
from blokus.engine import State, is_legal
import random


def test_population_and_elites():
    std=np.ones(len(ACTIVE));p=population(INITIAL,std,INITIAL,np.random.default_rng(7),10)
    assert len(p)==10 and all(x[0]==12 and x[9]==x[10]==0 for x in p)
    assert np.allclose(p[2][ACTIVE]+p[3][ACTIVE],2*INITIAL[ACTIVE])
    mean,newstd,best=update_distribution(p,np.arange(10),INITIAL,std)
    assert np.array_equal(best,p[9]) and np.all(newstd>=.25)


def test_linear_agent_and_resume(tmp_path):
    checkpoint=tmp_path/'agent.json';linear_checkpoint(checkpoint,INITIAL,{})
    a=Agent(str(checkpoint));b=Agent('heuristic');s=State()
    assert a.act(s,random.Random(10))==b.act(s,random.Random(10))
    assert is_legal(s,a.act(s,random.Random(5)))
    args=dict(baseline='artifacts/best-trained.pt',seconds=60,count=4,prefix_count=1,seed=1234)
    train(tmp_path/'full',generations=2,**args)
    train(tmp_path/'split',generations=1,**args)
    train(tmp_path/'resume',generations=1,resume=tmp_path/'split/cem-state.json',**args)
    full=json.loads((tmp_path/'full/cem-state.json').read_text());res=json.loads((tmp_path/'resume/cem-state.json').read_text())
    for k in ['generation','mean','std','incumbent','rng','games','evaluations']:assert full[k]==res[k]


def test_linear_search_does_not_change_state_or_weights(tmp_path):
    path=tmp_path/'linear.json';linear_checkpoint(path,INITIAL,{})
    before=path.read_bytes();agent=Agent(str(path)+':search');s=State()
    move=agent.act(s,random.Random(7))
    assert is_legal(s,move) and s==State()
    assert 0<agent.stats['nodes']<=16 and path.read_bytes()==before
