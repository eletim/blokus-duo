import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
import numpy as np
import torch
from h_distill_train import cache,batch,loss,Trainer
from blokus.ppo_policy import PPOAgent,collate
from blokus.engine import State,legal_moves
from blokus.ppo_value_leaf import value_only

def rows():
 qs=legal_moves(State());q=qs[0].move
 return [dict(moves=[],targets=[[q.piece,q.orientation,q.row,q.col]])]

def test_cache_mask_ties_and_policy_only_value():
 torch.set_num_threads(1);p=PPOAgent(1751401,'cpu');r=rows();q=legal_moves(State())[1].move;r[0]['targets'].append([q.piece,q.orientation,q.row,q.col]);data=cache(p,r)
 assert data[0][3].sum()==1 and (data[0][3]>0).sum()==2
 obs,_=p._observation(State());before=value_only(p,State());t=Trainer(1751401,data,'cpu');t.step();p.model.policy.load_state_dict(t.head.state_dict())
 assert value_only(p,State())==before
 padded=batch(data,[0],'cpu');assert padded[-1].all();total,ce,kl,l=loss(t.head,padded);assert torch.isfinite(total)
 assert not torch.equal(t.head[0].weight,t.agent.model.policy[0].weight)


def test_resume_exact_cpu(tmp_path):
 torch.set_num_threads(1);data=cache(PPOAgent(1751401,'cpu'),rows());a=Trainer(1751401,data,'cpu');a.step();a.save(tmp_path/'resume.pt');a.step()
 b=Trainer(1751401,data,'cpu');b.load(tmp_path/'resume.pt');b.step()
 assert all(torch.equal(v,b.head.state_dict()[k]) for k,v in a.head.state_dict().items())
 assert a.rng.bit_generator.state==b.rng.bit_generator.state

def test_single_legal_choice_zero_gradient_is_valid():
 torch.set_num_threads(1)
 p=PPOAgent(1751401,'cpu');r=rows();data=cache(p,r)
 x,prior,old,target,v=data[0]
 singleton=[(x[:1],prior[:1],old[:1],torch.ones(1),v)]
 t=Trainer(1751401,singleton,'cpu');metrics=t.step()
 assert metrics['zero_gradient'] and metrics['gradient_norm']==0 and t.updates==1
