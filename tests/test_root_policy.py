import random
import numpy as np
import torch
import pytest
from blokus.root_policy import RootPolicy,inputs,masked_loss,StudentAgent
from blokus.engine import State,legal_moves,is_legal
from blokus.ai import features,heuristic
from blokus.value_search import BudgetAgent
from blokus.research import clone

def test_prior_initial_and_input_bonus():
    s=State(remaining=[0,3],last=[0,None],turn=1);qs=legal_moves(s);x,st=inputs(s,qs)
    t=clone(s);t.last[0]=1;y,tt=inputs(t,qs)
    assert not np.array_equal(st,tt) and np.array_equal(x,y)
    net=RootPolicy().eval()
    assert torch.equal(net(torch.tensor(x),torch.tensor(st)),torch.zeros(len(qs)))
    assert all(is_legal(s,q.move) for q in qs)

def test_mask_unknown_never_contributes_and_bad_label_rejected():
    logits=torch.tensor([[1.,2.,1e8]],requires_grad=True);mask=torch.tensor([[True,True,False]])
    loss=masked_loss(logits,mask,torch.tensor([1]));loss.backward();assert logits.grad[0,2]==0
    with pytest.raises(ValueError):masked_loss(logits,mask,torch.tensor([2]))

def test_heuristic_student_same_completed_search_and_pass():
    s=State(remaining=[3,3]);a=StudentAgent(None,1000,device='cpu');b=BudgetAgent('value',None,.5,24,1000,'cpu')
    assert a.act(s,random.Random(4))==b.act(s,random.Random(4))
    assert a.roots==b.roots and a.nodes==b.nodes
    assert a.act(State(remaining=[0,1]),random.Random(2)) is None

def test_input_padded_network_and_shared_initialization():
    torch.manual_seed(900);a=RootPolicy();torch.manual_seed(900);b=RootPolicy()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=torch.zeros(2,128,39);st=torch.zeros(2,49);assert a(x,st).shape==(2,128)
