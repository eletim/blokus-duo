import copy
import numpy as np
import pytest
import torch
from blokus.ppo_policy import TerminalNet,collate,observe
from blokus.ppo_ragged import CompactBatch,forward
from blokus.ppo_runtime import seed_all
from blokus.engine import State

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('channels',[4,6])
def test_compact_order_pass_mask_state_mean(device,channels):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 seed_all(178100);a=TerminalNet(channels).to(device)
 with torch.no_grad():a.policy[-1].weight.normal_(0,.1)
 b=copy.deepcopy(a);obs,_=observe(State(),channels=channels)
 # Unequal candidate counts, including pass placeholder. Mean is over states.
 small=(obs[0].copy(),obs[1].copy(),obs[2][:3].copy(),obs[3][:3].copy())
 passed=(obs[0].copy(),obs[1].copy(),np.zeros((1,39),np.float32),np.full((1,5),196,np.int64))
 xs=[obs,small,passed];before=copy.deepcopy(xs)
 la,va=a(*collate(xs,device));lb,vb=forward(b,CompactBatch(xs,device));mask=torch.isfinite(la)
 assert torch.equal(mask,torch.isfinite(lb));assert torch.allclose(la[mask],lb[mask],atol=2e-5,rtol=0);assert torch.equal(va,vb)
 da=torch.distributions.Categorical(logits=la);db=torch.distributions.Categorical(logits=lb);ix=torch.tensor([0,2,0],device=device)
 loss_a=(-da.log_prob(ix)-.01*da.entropy()+va.square()).mean();loss_b=(-db.log_prob(ix)-.01*db.entropy()+vb.square()).mean()
 assert torch.allclose(loss_a,loss_b,atol=2e-5,rtol=0);loss_a.backward();loss_b.backward()
 assert all(torch.allclose(x.grad,y.grad,atol=2e-5,rtol=0) for x,y in zip(a.parameters(),b.parameters()))
 assert db.entropy()[2].item()==0 and db.log_prob(ix)[2].item()==0
 assert all(np.array_equal(x,y) for o,p in zip(xs,before) for x,y in zip(o,p))
 assert torch.equal(a.position.weight.grad[196],torch.zeros_like(a.position.weight.grad[196]))

def test_empty_frontier_starts_unchanged():
 obs,_=observe(State(),channels=4)
 assert obs[0].shape==(4,14,14)
 assert obs[0][2].sum()==obs[0][3].sum()==1


def test_compact_resident_invalid():
 from blokus.ppo_gpu import update_fast
 m=TerminalNet(4)
 with pytest.raises(ValueError):update_fast(m,torch.optim.Adam(m.parameters()),[],packed=True,compact=True)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_compact_kl_and_nonfinite_before_step(device):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 from blokus.ppo_gpu import update_fast
 seed_all(178123);m=TerminalNet(4).to(device);obs,_=observe(State(),channels=4)
 with torch.no_grad():
  l,v=m(*collate([obs],device));lp=torch.distributions.Categorical(logits=l).log_prob(torch.tensor([0],device=device)).item()
 rec=[dict(obs=obs,action=0,log_prob=lp-2,value=0.,**{'return':float(i%2)}) for i in range(16)]
 o=torch.optim.Adam(m.parameters(),lr=.0003);u=update_fast(m,o,rec,compact=True)
 assert u['early_kl'] and u['updates']==1
 before=copy.deepcopy(m.state_dict());rec[0]['return']=float('nan');old=copy.deepcopy(o.state_dict())
 with pytest.raises(FloatingPointError):update_fast(m,o,rec,compact=True)
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
 from scripts.bench_ppo_runtime import equal
 assert equal(old,o.state_dict())
