import queue
import threading
import numpy as np
import pytest
import torch
from blokus.batched_inference import InferenceClient, BatchServer
from blokus.engine import State
from blokus.ppo_policy import PPOAgent,collate
from blokus.ppo_value_leaf import value_inputs,value_only


def test_reply_identity_and_error():
    requests=queue.Queue();replies=queue.Queue();c=InferenceClient(3,1751401,requests,replies)
    replies.put((2,None,None))
    with pytest.raises(RuntimeError,match='stale'):c.call('value',(1,))
    assert requests.get()[:4]==(3,1,1751401,'value')
    replies.put((2,None,'failed backend'))
    with pytest.raises(RuntimeError,match='failed backend'):c.call('value',(1,))


def test_terminal_never_sends_request():
    q=queue.Queue();c=InferenceClient(0,1751401,q,queue.Queue());s=State();s.passes=2
    assert c.leaf(s,4)==value_only(None,s)==0
    assert q.empty()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA integration')
def test_batched_mixed_requests_and_state_immutable():
    torch.set_num_threads(1)
    p=PPOAgent(1751401,'cuda');s=State();before=repr(s)
    obs,_=p._observation(s);rq=queue.Queue();responses=[queue.Queue() for _ in range(3)]
    rq.put((0,1,1751401,'policy',obs));rq.put((1,7,1751401,'policy',obs));rq.put((2,5,1751401,'value',value_inputs(s,4)))
    server=BatchServer({1751401:p.model},rq,responses);assert server.step()
    with torch.inference_mode():l,v=p.model(*collate([obs],'cuda'))
    for i,serial in [(0,1),(1,7)]:
        n,(actual,value),error=responses[i].get();assert n==serial and error is None
        np.testing.assert_allclose(actual,l[0].cpu().numpy(),atol=2e-5,rtol=0)
        assert abs(value-v.item())<=2e-5
    serial,value,error=responses[2].get();assert serial==5 and error is None
    assert abs(value-value_only(p,s))<=2e-5
    assert repr(s)==before
    assert server.summary()['batch_histogram']=={'policy:2':1,'value:1':1}


def test_transport_retains_search_cache_semantics():
    import copy,random
    from blokus.ppo_search_deeper import DeepSearch
    from blokus.engine import legal_moves,play
    torch.set_num_threads(1)
    p=PPOAgent(1751401,'cpu');q=copy.copy(p)
    q.value_backend=lambda s:value_only(p,s)
    s=State();rng=random.Random(733)
    for _ in range(24):
        moves=legal_moves(s);play(s,rng.choice(moves).move if moves else None)
    a=DeepSearch(p,60000,max_depth=4);b=DeepSearch(q,60000,max_depth=4)
    assert a.act(s,random.Random(2))==b.act(s,random.Random(2))
    for k in ('depth','nodes','nn_policy_calls','nn_value_calls','cache_hits','root_scores'):
        assert a.stats[k]==b.stats[k],k
