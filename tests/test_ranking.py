import random
import numpy as np
import torch
from blokus.engine import State,legal_moves,play
from blokus.research import clone
from blokus.ranking_data import candidate_leaves,continuation,initialize_worker
from blokus.ranking_train import losses,differences,train,dataset
from blokus.value_search import encode


def test_pair_difference_shift_permutation_and_gradients():
    p=torch.tensor([[.1,.5,-.3,.2]],requires_grad=True);y=torch.tensor([[.2,.3,-.4,.1]])
    assert torch.allclose(differences(p+3),differences(p),atol=1e-6)
    loss,parts=losses(p,y);permutation=[2,0,3,1]
    assert torch.allclose(loss,losses(p[:,permutation],y[:,permutation])[0])
    loss.backward();assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    # Uniform shifts affect only the absolute anchor.
    _,shifted=losses(p+1,y+1)
    assert torch.allclose(parts['diff'],shifted['diff'],atol=1e-6)


def test_candidates_and_shared_continuations_are_pure():
    initialize_worker();s=State();rng=random.Random(4)
    for _ in range(18):
        qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None)
    before=clone(s);leaves,moves=candidate_leaves(s,rng)
    assert s==before and len(leaves)==4 and len({tuple(m) for m in moves})==4
    from blokus.ranking_data import AGENTS
    for leaf in leaves:
        assert leaf.turn==s.turn;old=clone(leaf)
        assert continuation(leaf,AGENTS[0],44)==continuation(leaf,AGENTS[0],44)
        assert leaf==old


def fixture(path):
    s=State();b,x=encode(s)
    np.savez(path,board=np.tile(np.packbits(b.reshape(-1)),(20,4,1)),state=np.tile(x,(20,4,1)),
             returns=np.random.default_rng(88).normal(0,.2,(20,4,2)).astype(np.float32),game=np.arange(1,21))


def test_group_split_and_exact_resume(tmp_path):
    path=tmp_path/'data.npz';fixture(path);_,_,_,ti,vi=dataset(path,'cpu')
    assert not set(ti.tolist())&set(vi.tolist()) and len(vi)==2
    args=dict(path=path,seed=301,seconds=30,device='cpu')
    train(output=tmp_path/'full',steps=4,**args);train(output=tmp_path/'part',steps=2,**args)
    train(output=tmp_path/'resumed',steps=2,resume=tmp_path/'part/latest.pt',**args)
    a=torch.load(tmp_path/'full/latest.pt',weights_only=False);b=torch.load(tmp_path/'resumed/latest.pt',weights_only=False)
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    assert torch.equal(a['torch_rng'],b['torch_rng']) and a['python_rng']==b['python_rng']
    for k,vs in a['optimizer']['state'].items():assert all(torch.equal(v,b['optimizer']['state'][k][n]) for n,v in vs.items())


def test_budget_recovers_after_one_long_stall(monkeypatch):
    import blokus.value_search as module
    clock=[0.];first=[True]
    def now():clock[0]+=.00001;return clock[0]
    monkeypatch.setattr(module.time,'perf_counter',now)
    a=module.BudgetAgent(budget_ms=6,device='cpu');s=State()
    def evaluate(states):
        if first[0]:clock[0]+=.1;first[0]=False
        return np.zeros(len(states))
    a.evaluator.batch=evaluate
    a.act(s,random.Random(1));assert a.roots==0 and a.stats()['overruns']==1
    a.act(s,random.Random(2));assert a.roots>0
