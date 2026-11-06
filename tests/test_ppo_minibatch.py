import pytest
from blokus.ppo_gpu import update_fast
@pytest.mark.parametrize('batch',[0,-1,1.5])
def test_invalid_minibatch(batch):
 with pytest.raises(ValueError):update_fast(None,None,[],minibatch=batch)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_minibatch64_changes_policy_with_finite_gradients(device):
 import copy,torch,numpy as np
 from blokus import ppo_runtime as r
 from blokus.ppo_policy import TerminalNet
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 r.seed_all(176007);model=TerminalNet(4).to(device);records,_,_=r.collect(model,copy.deepcopy(model),0,176007,backend='native' if r.library() else 'python');before=copy.deepcopy(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=.0003)
 result=update_fast(model,opt,records,packed=True,minibatch=64)
 assert np.isfinite([result[k] for k in ('loss','grad_norm','approx_kl','entropy')]).all()
 assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
 assert all(int(s['step'])==result['updates'] for s in opt.state.values())

def test_resume_rejects_silent_batch_change(tmp_path,monkeypatch):
 from blokus import ppo_minibatch as m
 monkeypatch.setattr(m.r,'seed_all',lambda x:None)
 monkeypatch.setattr(m.r,'restore',lambda *a:(None,None,None,dict(iteration=4097,seed=1,channels=4,minibatch=16)))
 with pytest.raises(AssertionError):m.run(tmp_path,'checkpoint',4098,64,'exclusions')
