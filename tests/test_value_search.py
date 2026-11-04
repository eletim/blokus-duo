import random
import numpy as np
import pytest
import torch
from blokus.engine import State,bit,legal_moves,play,is_legal
from blokus.ai import seed_all
from blokus.research import clone
from blokus.value_search import *
from blokus.value_train import train


def test_outcome_encoding_and_antisymmetry():
    seed_all(1);s=State(boards=[bit((4,4)),bit((9,9))]);s.remaining[0]=0;s.last[0]=0
    b,x=encode(s);assert b.shape==(6,14,14) and x.shape==(49,)
    s.last[0]=1;assert encode(s)[1][46]!=x[46]
    model=OutcomeNet();b=torch.tensor(b[None],dtype=torch.float32);x=torch.tensor(x[None])
    ob,ox=swap_inputs(b,x)
    assert torch.allclose(model(b,x),-model(ob,ox),atol=1e-6)
    loss=(model(b,x)-.3).square().mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_value_terminal_pass_and_perspective():
    s=State(remaining=[0,1],last=[0,None],passes=2)
    assert ValueEvaluator(device='cpu').batch([s])[0]==21/40
    s.turn=1;assert ValueEvaluator(device='cpu').batch([s])[0]==-21/40
    s=State(remaining=[0,0],last=[0,1],passes=1)
    assert exact_terminal(s) and ValueEvaluator(device='cpu').batch([s])[0]==5/40
    s=State(remaining=[0,1]);a=BudgetAgent(device='cpu')
    assert a.act(s,random.Random(1)) is None
    with pytest.raises(ValueError):a.act(State(passes=2),random.Random(1))


def test_budget_search_legal_pure_and_measured():
    a=BudgetAgent(budget_ms=6,device='cpu');s=State();before=clone(s)
    m=a.act(s,random.Random(1));assert is_legal(s,m) and s==before
    stat=a.stats();assert stat['allowance_ms']==6 and stat['calls']==1 and stat['nodes']>=0
    assert stat['total_seconds']>0


def test_late_child_is_discarded(monkeypatch):
    import blokus.value_search as module
    clock=[0.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    a=BudgetAgent(budget_ms=6,device='cpu');s=State()
    qs=legal_moves(s);x,_=features(s,qs);fallback=qs[int(np.argsort(-heuristic(x),kind='stable')[0])].move
    def late(states):clock[0]+=.007;return np.ones(len(states))*100
    a.evaluator.batch=late
    assert a.act(s,random.Random(2))==fallback
    assert a.stats()['completed_roots']==0 and a.stats()['overruns']==1


def test_outcome_training_exact_resume(tmp_path):
    s=State();b=[];x=[]
    for i in range(20):
        eb,ex=encode(s);b.append(np.packbits(eb.reshape(-1)));x.append(ex)
        qs=legal_moves(s);play(s,qs[0].move if qs else None,validate=False)
    path=tmp_path/'dataset.npz'
    np.savez(path,board=np.array(b),state=np.array(x),target=np.linspace(-.2,.2,20,dtype=np.float32),kind=np.zeros(20,dtype=np.uint8),game=np.arange(1,21))
    args=dict(paths=[path],seconds=30,seed=88,device='cpu')
    train(output=tmp_path/'full',steps=4,**args)
    train(output=tmp_path/'split',steps=2,**args)
    train(output=tmp_path/'resumed',steps=2,resume=tmp_path/'split/latest.pt',**args)
    a=torch.load(tmp_path/'full/latest.pt',weights_only=False);c=torch.load(tmp_path/'resumed/latest.pt',weights_only=False)
    assert a['step']==c['step']==4 and torch.equal(a['torch_rng'],c['torch_rng'])
    assert all(torch.equal(v,c['model'][k]) for k,v in a['model'].items())
    for k,vals in a['optimizer']['state'].items():
        assert all(torch.equal(v,c['optimizer']['state'][k][n]) for n,v in vals.items())


def test_pooled_architecture_and_board_symmetries():
    from blokus.engine import mask,cells_of
    s=State();rng=random.Random(99)
    for _ in range(8):play(s,rng.choice(legal_moves(s)).move)
    board,st=encode(s);other=clone(s)
    other.boards=[mask({(c,r) for r,c in cells_of(b)}) for b in s.boards]
    b2,x2=encode(other)
    assert np.array_equal(b2,board.transpose(0,2,1)) and np.array_equal(x2,st)
    flat=OutcomeNet();small=OutcomeNet('pooled')
    assert sum(p.numel() for p in small.parameters())<sum(p.numel() for p in flat.parameters())/5
    x=torch.tensor(st[None]);b=torch.tensor(board[None],dtype=torch.float32);ob,ox=swap_inputs(b,x)
    assert torch.allclose(small(b,x),-small(ob,ox),atol=1e-6)


def test_batched_search_late_result_and_purity(monkeypatch):
    import blokus.value_search as module
    clock=[0.]
    def ticking():clock[0]+=.00005;return clock[0]
    monkeypatch.setattr(module.time,'perf_counter',ticking)
    a=BudgetAgent(device='cpu',scheduler='batched');s=State();before=clone(s)
    def late(states):clock[0]+=.010;return np.ones(len(states))*10
    a.evaluator.batch=late;m=a.act(s,random.Random(3))
    assert s==before and is_legal(s,m) and a.stats()['completed_roots']==0
    assert a.stats()['overruns']==1
