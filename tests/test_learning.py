import random
import numpy as np
import torch
from blokus.ai import *

def test_policy_gradient_and_resume(tmp_path):
    seed_all(7);m=Net();opt=torch.optim.Adam(m.parameters());s=State();qs=legal_moves(s)
    x,st=features(s,qs);logits,v=m(*tensors(x,st,'cpu'))
    before={k:v.clone() for k,v in m.state_dict().items()}
    (-logits.log_softmax(0)[0]+(v-.5).square()).backward();opt.step()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    assert any(not torch.equal(before[k],v) for k,v in m.state_dict().items())
    path=tmp_path/'test.pt';save(path,m,opt,4,'selfplay')
    expected=(random.random(),np.random.rand(),torch.rand(1))
    other=Net();otheropt=torch.optim.Adam(other.parameters());ck=load(path,other,otheropt,True)
    assert ck['episode']==4 and otheropt.state_dict()['state']
    assert random.random()==expected[0] and np.random.rand()==expected[1] and torch.equal(torch.rand(1),expected[2])
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in m.state_dict().items())
