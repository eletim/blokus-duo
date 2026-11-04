import random
import numpy as np
import torch
from blokus.engine import State,legal_moves,play
from blokus.value_search import OutcomeNet,ValueEvaluator,encode,exact_terminal,score_value
from blokus.visited_data import state_hash,stratified_subset,state_dict
from blokus.visited_train import live_values,decision_metrics,train


def fixture(path):
    rng=np.random.default_rng(66);b,x=encode(State())
    np.savez(path,board=np.tile(np.packbits(b.reshape(-1)),(12,4,3,1)),state=np.tile(x,(12,4,3,1)),returns=rng.normal(0,.2,(12,4,3,4)).astype(np.float32),mask=np.ones((12,4,3),bool),terminal=np.zeros((12,4,3),bool),score=np.zeros((12,4,3),np.float32),game=np.arange(12))


def test_live_criterion_matches_actual_evaluator(tmp_path):
    from blokus.ai import seed_all
    seed_all(9);model=OutcomeNet('pooled')
    with torch.no_grad():model.head[-1].weight.normal_(std=.1)
    path=tmp_path/'model.pt';torch.save(dict(format=4,arch='pooled',model=model.state_dict()),path)
    s=State();rng=random.Random(7)
    for _ in range(20):
        qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None)
    states=[s,State(remaining=[0,0],last=[0,1],passes=2)]
    bx=[encode(s) for s in states];b=torch.tensor(np.array([v[0] for v in bx]),dtype=torch.float32);x=torch.tensor(np.array([v[1] for v in bx]))
    p=model(b,x);live=live_values(p,x,torch.tensor([exact_terminal(s) for s in states]),torch.tensor([score_value(s) for s in states]))
    actual=ValueEvaluator(path,'cpu',.5,42).batch(states)
    assert np.allclose(live.detach().numpy(),actual,atol=1e-6)


def test_hash_semantics_and_tie_regret():
    a=State();b=State();a.last[0]=0;b.last[0]=1
    assert state_hash(a)==state_hash(b)
    a.remaining[0]=0;b.remaining[0]=0;assert state_hash(a)!=state_hash(b)
    p=torch.zeros(1,4,3);y=torch.tensor([[[0.]*3,[.1]*3,[.2]*3,[.3]*3]]);m=torch.ones_like(p,dtype=torch.bool)
    assert abs(decision_metrics(p,y,m)['decision_regret_points']-6)<1e-5


def test_dual_selector_resume(tmp_path):
    path=tmp_path/'data.npz';fixture(path);args=dict(path=path,dev_path=path,seed=601,seconds=30,device='cpu')
    train(output=tmp_path/'full',steps=4,**args);train(output=tmp_path/'part',steps=2,**args)
    train(output=tmp_path/'resumed',steps=2,resume=tmp_path/'part/latest.pt',**args)
    a=torch.load(tmp_path/'full/latest.pt',weights_only=False);b=torch.load(tmp_path/'resumed/latest.pt',weights_only=False)
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items()) and torch.equal(a['torch_rng'],b['torch_rng'])
    for k,vs in a['optimizer']['state'].items():assert all(torch.equal(v,b['optimizer']['state'][k][n]) for n,v in vs.items())
