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
