import gzip,json
import pytest
from blokus.ppo_scale import history
from blokus.research import sha
from blokus.ppo_curve_eval import Stochastic
from blokus.terminal_ppo import TerminalAgent

def test_external_history_hash(tmp_path):
 p=tmp_path/'history.json.gz';p.write_bytes(gzip.compress(json.dumps([{'id':1}]).encode()));c={'history':p.name,'history_sha256':sha(p)}
 assert history(c,tmp_path/'ck.pt')==[{'id':1}]
 p.write_bytes(gzip.compress(b'[]'))
 with pytest.raises(AssertionError):history(c,tmp_path/'ck.pt')

def test_legacy_history():
 assert history({'games_gzip':gzip.compress(b'[{"id":2}]')},'unused')==[{'id':2}]

def test_stochastic_eval_uses_supplied_rng_and_legal_candidates(tmp_path):
 import random
 import torch
 from blokus.terminal_ppo import TerminalNet
 from blokus.engine import State,is_legal
 torch.set_num_threads(1);torch.manual_seed(1400);path=tmp_path/'net.pt';torch.save({'model':TerminalNet().state_dict()},path)
 agent=Stochastic(path,device='cpu');a=random.Random(42);b=random.Random(42);state=State()
 left=[agent.act(state,a) for _ in range(12)];right=[agent.act(state,b) for _ in range(12)]
 assert left==right and all(is_legal(state,m) for m in left)
 assert len(set(left))>1
 assert Stochastic.act is not TerminalAgent.act


def test_continuation_never_overwrites_existing_output(tmp_path):
 from blokus.ppo_scale import train
 marker=tmp_path/'config.json';marker.write_text('preserve')
 with pytest.raises(FileExistsError):train(1401,'unused',tmp_path)
 assert marker.read_text()=='preserve'

def test_history_chain_and_missing_segment(tmp_path):
 p=tmp_path/'part.json.gz';p.write_bytes(gzip.compress(b'[{"id":3}]'))
 c={'history_chain':[dict(path=p.name,sha256=sha(p),games=1)]}
 assert history(c,tmp_path/'ck.pt')==[{'id':3}]
 p.unlink()
 with pytest.raises(FileNotFoundError):history(c,tmp_path/'ck.pt')

def test_prefix_exclusion_does_not_change_other_games(monkeypatch):
 from blokus import terminal_ppo as p
 from blokus.tournament import prefix_state
 from blokus.visited_data import state_hash
 monkeypatch.setattr(p,'sample_batch',lambda model,obs:([0]*len(obs),[0.]*len(obs),[0.]*len(obs)))
 _,first,_=p.collect(None,None,0,1401,8)
 banned={state_hash(prefix_state({'moves':first[0]['prefix']}))}
 _,second,_=p.collect(None,None,0,1401,8,forbidden_starts=banned)
 assert second[0]['prefix_rejections']>=1 and second[0]['prefix']!=first[0]['prefix']
 assert all(state_hash(prefix_state({'moves':g['prefix']})) not in banned for g in second)
 assert [g['prefix'] for g in first[1:]]==[g['prefix'] for g in second[1:]]


def test_incremental_state_index_union_and_integrity(tmp_path):
 from blokus.ppo_curve_eval import read_hash_index
 from blokus.value_experiment import write
 a='01'*32;b='02'*32
 base=tmp_path/'base';base.mkdir();(base/'0.gz').write_bytes(gzip.compress(bytes.fromhex(a)))
 write(base/'index.json',dict(count=1,parts=[dict(path='0.gz',sha256=sha(base/'0.gz'))]))
 delta=tmp_path/'delta';delta.mkdir();(delta/'0.gz').write_bytes(gzip.compress(bytes.fromhex(b)))
 write(delta/'index.json',dict(count=2,bases=[dict(path='../base/index.json',sha256=sha(base/'index.json'))],parts=[dict(path='0.gz',sha256=sha(delta/'0.gz'))]))
 assert read_hash_index(delta/'index.json')=={a,b}
 (base/'0.gz').write_bytes(b'broken')
 with pytest.raises(AssertionError):read_hash_index(delta/'index.json')
