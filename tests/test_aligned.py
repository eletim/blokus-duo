import random
import numpy as np
import torch
from blokus.engine import State,legal_moves,play
from blokus.research import clone
from blokus.aligned_data import replies_group,initialize,finish,ply_seed
from blokus.aligned_train import candidate_values,objective,train
from blokus.value_search import encode


def fixture(path):
    b,x=encode(State());rng=np.random.default_rng(44)
    np.savez(path,board=np.tile(np.packbits(b.reshape(-1)),(20,4,3,1)),state=np.tile(x,(20,4,3,1)),
             returns=rng.normal(0,.2,(20,4,3,4)).astype(np.float32),mask=np.ones((20,4,3),dtype=bool),game=np.arange(1,21))


def test_reply_min_mask_and_gradients():
    p=torch.tensor([[[1.,2.,-100.],[3.,4.,5.],[2.,1.,0.],[1.,2.,3.]]],requires_grad=True)
    mask=torch.ones_like(p,dtype=torch.bool);mask[0,0,2]=False
    assert candidate_values(p,mask).tolist()==[[1.,3.,0.,1.]]
    for mode in ['absolute','difference','ranking']:
        loss=objective(p,torch.zeros_like(p),mask,mode);assert torch.isfinite(loss)
        grad=torch.autograd.grad(loss,p,retain_graph=True)[0];assert grad[0,0,2]==0 and torch.isfinite(grad).all()


def test_group_reply_order_and_common_random_replay():
    from blokus.ai import features,heuristic
    initialize();s=State();rng=random.Random(77);eligible=[]
    for _ in range(22):
        qs=legal_moves(s)
        if len(qs)>=4:eligible.append(clone(s))
        play(s,rng.choice(qs).move if qs else None)
    s=eligible[-1];old=clone(s);leaves,moves,replies=replies_group(s)
    for k,move in enumerate(moves):
        from blokus.engine import Move
        child=clone(s);play(child,Move(*move));qs=legal_moves(child)
        if qs:
            x,_=features(child,qs);top=np.argsort(-heuristic(x),kind='stable')[:3]
            expected=[[qs[int(i)].move.piece,qs[int(i)].move.orientation,qs[int(i)].move.row,qs[int(i)].move.col] for i in top]
            assert replies[k]==expected
        else:assert replies[k]==[None]
        assert all(leaf.turn==s.turn for leaf in leaves[k])
    assert s==old and ply_seed(99,4,0)==ply_seed(99,4,0)
    leaf=leaves[0][0];before=clone(leaf)
    assert finish(leaf,55,2)==finish(leaf,55,2) and leaf==before


def test_aligned_resume(tmp_path):
    path=tmp_path/'data.npz';fixture(path)
    args=dict(path=path,mode='ranking',seed=501,seconds=30,device='cpu')
    train(output=tmp_path/'full',steps=4,**args);train(output=tmp_path/'part',steps=2,**args)
    train(output=tmp_path/'resumed',steps=2,resume=tmp_path/'part/latest.pt',**args)
    a=torch.load(tmp_path/'full/latest.pt',weights_only=False);b=torch.load(tmp_path/'resumed/latest.pt',weights_only=False)
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items()) and torch.equal(a['torch_rng'],b['torch_rng'])
    for k,vs in a['optimizer']['state'].items():assert all(torch.equal(v,b['optimizer']['state'][k][n]) for n,v in vs.items())
