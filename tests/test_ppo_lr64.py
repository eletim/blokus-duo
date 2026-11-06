import pytest
from blokus import ppo_lr64 as m
@pytest.mark.parametrize('condition,lr',[('A',.0003),('B',.0003),('C',.0006)])
def test_explicit_lr_and_metadata(tmp_path,monkeypatch,condition,lr):
 # Reject a corrupted optimizer before collection or any mutation.
 class Opt: param_groups=[{'lr':.009}]
 mb=m.CONDITIONS[condition][0]
 ck=dict(seed=1751401,channels=4,iteration=4100,condition=condition,minibatch=mb,learning_rate=lr,series_seed=1771401)
 monkeypatch.setattr(m.r,'seed_all',lambda _:None);monkeypatch.setattr(m.r,'restore',lambda *a:(None,None,Opt(),ck))
 with pytest.raises(AssertionError,match='optimizer LR'):m.run(tmp_path,'fake',4101,condition,'fake')

def test_reject_other_condition(tmp_path,monkeypatch):
 ck=dict(seed=1751401,channels=4,iteration=4100,condition='B',minibatch=64,learning_rate=.0003,series_seed=1771401)
 monkeypatch.setattr(m.r,'seed_all',lambda _:None);monkeypatch.setattr(m.r,'restore',lambda *a:(None,None,None,ck))
 with pytest.raises(AssertionError):m.run(tmp_path,'fake',4101,'C','fake')

def test_collapse_monitor_survives_checkpoint_boundary(tmp_path,monkeypatch):
 import json
 class Opt: param_groups=[{'lr':.0006}]
 ck=dict(seed=1751401,channels=4,iteration=4096,condition='C',minibatch=64,learning_rate=.0006,series_seed=1771401,logs=[dict(update=dict(entropy=.01,approx_kl=.001)) for _ in range(63)])
 monkeypatch.setattr(m.r,'seed_all',lambda _:None);monkeypatch.setattr(m.r,'restore',lambda *a:(None,None,Opt(),ck));monkeypatch.setattr(m,'sha',lambda _: 'hash')
 games=[dict(opponent=k,scores=[0,0],learner_seat=0) for k in ('current','past','heuristic')]
 monkeypatch.setattr(m.r,'collect',lambda *a,**kw:([],games,{}));monkeypatch.setattr(m,'update_fast',lambda *a,**kw:dict(entropy=.01,approx_kl=.001))
 excluded=tmp_path/'excluded.json';excluded.write_text(json.dumps(dict(states=[])))
 with pytest.raises(RuntimeError,match='collapse'):m.run(tmp_path,'fake',4097,'C',excluded)
