import sys,json,random
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus import terminal_ppo as old,ppo_policy as new
from blokus.engine import State,legal_moves,play
from blokus.ai import seed_all
from blokus.research import sha
seed_all(1);states=[];rng=random.Random(1580000)
for game in range(4):
 s=State()
 for ply in range(35):
  if s.done:break
  if ply%5==0:
   from blokus.research import clone
   states.append(clone(s))
  qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None)
checks=[]
for seed in (1401,1402,1403):
 path=json.loads(Path('artifacts/ppo-generation2/lock.json').read_text())['models'][str(seed)]['path']
 for device in ('cpu','cuda'):
  a=old.TerminalAgent(path,device);b=new.PPOAgent(seed,device,generation="gen2")
  for s in states:
   x,qs=old.observe(s);y,rs=new.observe(s);assert qs==rs and all(np.array_equal(i,j) for i,j in zip(x,y))
   with torch.no_grad():av=a.model(*old.collate([x],device));bv=b.model(*new.collate([y],device))
   assert all(torch.equal(i,j) for i,j in zip(av,bv));assert a.act(s,random.Random(7))==b.act(s,random.Random(7))
  checks.append(dict(seed=seed,device=device,states=len(states),inputs_bit_equal=True,logits_value_bit_equal=True,greedy_equal=True))
Path('artifacts/ppo-gen2/export-check.json').write_text(json.dumps(dict(pass_=True,checks=checks),indent=2))
print('export equivalence PASS',checks)
