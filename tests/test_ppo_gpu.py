import copy
import numpy as np
import torch
import pytest
from blokus import ppo_runtime as r
from blokus.ppo_gpu import update_fast
from blokus.ppo_policy import TerminalNet,collate

@pytest.mark.parametrize("device",["cpu","cuda"])
def test_consolidated_update_exact(device):
 if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
 r.seed_all(1750600);m=TerminalNet(4).to(device);rec,games,_=r.collect(m,copy.deepcopy(m),0,1750600,backend='native' if r.library() else 'python');a=copy.deepcopy(m);b=copy.deepcopy(m);oa=torch.optim.Adam(a.parameters(),lr=.0003);ob=torch.optim.Adam(b.parameters(),lr=.0003)
 r.seed_all(100);x=r.update(a,oa,rec);r.seed_all(100);y=update_fast(b,ob,rec,packed=True)
 assert x['updates']==y['updates'] and all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
 assert all(np.isfinite(y[k]) for k in ['loss','grad_norm','approx_kl'])
 assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
 for key in ('loss','policy_loss','value_mse','entropy','approx_kl','grad_norm','clip_fraction'):assert x[key]==y[key]
 batch=collate([rec[0]['obs']],device)
 with torch.no_grad():
  xa=a(*batch);xb=b(*batch)
 assert all(torch.equal(p,q) for p,q in zip(xa,xb))

def test_nonfinite_does_not_step():
 r.seed_all(1750601);m=TerminalNet(4);rec,_,_=r.collect(m,copy.deepcopy(m),0,1750601,backend='native' if r.library() else 'python');rec[0]['return']=float('nan');opt=torch.optim.Adam(m.parameters(),lr=.0003);before=copy.deepcopy(m.state_dict())
 import pytest
 with pytest.raises(FloatingPointError):update_fast(m,opt,rec)
 assert not opt.state and all(torch.equal(before[k],v) for k,v in m.state_dict().items())

def test_stepwise_kl_stop_unchanged():
 r.seed_all(1750602);m=TerminalNet(4);rec,_,_=r.collect(m,copy.deepcopy(m),0,1750602,backend='native' if r.library() else 'python')
 for item in rec:item['log_prob']-=2.
 a=copy.deepcopy(m);b=copy.deepcopy(m);oa=torch.optim.Adam(a.parameters(),lr=.0003);ob=torch.optim.Adam(b.parameters(),lr=.0003)
 r.seed_all(101);x=r.update(a,oa,rec);r.seed_all(101);y=update_fast(b,ob,rec,packed=True)
 assert x['early_kl'] and y['early_kl'] and x['updates']==y['updates']==1
 assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
