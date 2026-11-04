import random
import numpy as np
import pytest
import torch
from blokus.engine import State, Move, bit, play, legal_moves, is_legal
from blokus.research import *
from blokus.tournament import make_prefixes, prefix_state, summarize, digest
from blokus.research_train import train


def test_board_side_to_move_and_state_bonus():
    s = State(boards=[bit((4,4)),bit((9,9))])
    b = encode_board(s)
    assert b.shape == (4,14,14) and b.dtype == np.float32
    assert b[0,4,4] == b[1,9,9] == b[2,4,4] == b[3,9,9] == 1
    s.turn = 1
    assert np.array_equal(encode_board(s)[0], b[1])
    s.remaining[0] = 0; s.last[0] = 0
    mono = pack(s,legal_moves(s))[1]
    s.last[0] = 1
    other = pack(s,legal_moves(s))[1]
    assert len(mono) == 49 and mono[-2] != other[-2]
    s.passes = 1
    assert pack(s,legal_moves(s))[1][-1] == .5


def test_mask_pass_and_terminal():
    s = State(); mask = action_mask(s)
    assert mask.sum() == 414 and not mask[PASS_ID]
    assert set(np.flatnonzero(mask)) == {ACTION_IDS[q.move] for q in legal_moves(s)}
    s.remaining[0] = 0
    assert action_mask(s).sum() == 1 and action_mask(s)[PASS_ID]
    s.passes = 2
    assert not action_mask(s).any()


def test_value_sign_and_terminal_bonus():
    s = State(remaining=[0,1],last=[0,None],passes=2)
    assert terminal_value(s) == 21/109 and static_value(s) == 21/109
    s.turn = 1
    assert terminal_value(s) == -21/109
    assert terminal_value(s,0) == 21/109


def test_search_purity_legality_and_bound():
    s = State(); before = clone(s)
    qs, selected, values, nodes = search(s,width=3,reply_width=2)
    assert s == before and nodes <= 3*(1+2)
    assert len(values) == len(selected) == 3
    assert all(is_legal(s,qs[i].move) for i in selected)
    # For width1, independently form exactly the chosen root/reply leaf.
    qs, selected, values, nodes = search(s,width=1,reply_width=1)
    child=clone(s);play(child,qs[selected[0]].move)
    rs=legal_moves(child);x,_=features(child,rs);r=int(np.argmax(heuristic(x)))
    leaf=clone(child);play(leaf,rs[r].move)
    assert values[0] == static_value(leaf)


def test_search_forced_pass_and_end():
    s=State(remaining=[0,0],last=[0,1],passes=1)
    before=clone(s);qs,ids,values,nodes=search(s)
    child=clone(s);play(child,None)
    assert nodes == 1 and not qs and not ids and values[0] == -terminal_value(child)
    assert s == before
    assert search(child)[3] == 0


@pytest.mark.parametrize('method',['A','B','C'])
def test_models_finite_gradients_and_legal_prior(method):
    seed_all(3);s=State();qs=legal_moves(s);data=pack(s,qs);model=PolicyValue(method)
    logits,value=model(*to_device(data,'cpu'))
    assert logits.shape == (414,) and value.ndim == 0
    assert torch.allclose(logits,torch.tensor(heuristic(data[2])),atol=1e-5)
    (-logits.log_softmax(0)[0]+value.square()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_prefix_reproducibility_and_cluster_bootstrap():
    a=make_prefixes(100,4);assert a==make_prefixes(100,4)
    assert a['sha256']==digest(a['prefixes'])
    assert not {p['id'] for p in a['prefixes']} & {p['id'] for p in make_prefixes(101,4)['prefixes']}
    for p in a['prefixes']:assert not prefix_state(p).done
    games=[dict(prefix_id=str(i),seat=j,difference=(10 if j==0 else -10)) for i in range(4) for j in (0,1)]
    result=summarize(games)
    assert result['difference_ci95']==[0,0] and result['point_rate_ci95']==[.5,.5]
    with pytest.raises(ValueError):summarize(games[:-1])


@pytest.mark.parametrize('method',['A','B','C'])
def test_training_exact_resume(tmp_path, method):
    baseline='artifacts/best-trained.pt'
    args=dict(method=method,baseline=baseline,seconds=60,seed=345,device='cpu')
    train(output=tmp_path/'whole',games=4,**args)
    train(output=tmp_path/'split',games=2,**args)
    train(output=tmp_path/'resumed',games=2,resume=tmp_path/'split/latest.pt',**args)
    a=torch.load(tmp_path/'whole/latest.pt',weights_only=False)
    b=torch.load(tmp_path/'resumed/latest.pt',weights_only=False)
    assert a['episode']==b['episode']==4
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    assert torch.equal(a['torch_rng'],b['torch_rng']) and a['python_rng']==b['python_rng']
