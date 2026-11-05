import random
import numpy as np
import pytest
import torch
from blokus.engine import State,legal_moves,is_legal,play
from blokus.ai import seed_all
from blokus.ppo_policy import PPOAgent,observe,collate
from blokus.ppo_play import human_move

def test_exported_three_seeds_cpu_without_history():
 seed_all(1)
 for seed in (1401,1402,1403):
  a=PPOAgent(seed,'cpu');s=State();rng=random.Random(17)
  for _ in range(6):play(s,a.act(s,rng))
  assert a.metadata['iteration']==2048 and 'history' not in a.metadata

def test_legal_mask_and_exact_cells():
 s=State();o,qs=observe(s);assert len(qs)==414
 for q,cs in zip(qs,o[3]):assert {int(x) for x in cs if x!=196}=={r*14+c for r,c in q.move.cells}
 small=(o[0],o[1],o[2][:1],o[3][:1]);batch=collate([o,small],'cpu');assert batch[-1][1].sum()==1
 a=PPOAgent(1401,'cpu')
 with torch.no_grad():logits,_=a.model(*batch)
 assert torch.isneginf(logits[1,1:]).all()

def test_human_input_bounds_and_forced_pass():
 answers=iter(['-1','99999','nonsense','0']);s=State();m=human_move(s,lambda _:next(answers),lambda _:None);assert m==legal_moves(s)[0].move and is_legal(s,m)
 s.remaining=[0,0];assert human_move(s,lambda _:pytest.fail('forced pass must not ask'),lambda _:None) is None

def test_cpu_device_explicit():
 assert PPOAgent(1401,'cpu').device=='cpu'
