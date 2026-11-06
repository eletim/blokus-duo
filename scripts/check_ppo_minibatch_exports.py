import json,random
from pathlib import Path
import torch
from blokus.ppo_minibatch import R,SEEDS,write
from blokus.ppo_policy import TerminalNet,collate,observe
from blokus.engine import State,play,legal_moves
from blokus.ai import seed_all
seed_all(176008);s=State();rng=random.Random(176008);states=[]
for i in range(32):
 if s.done:break
 if i%4==0:states.append(s.copy() if hasattr(s,'copy') else __import__('copy').deepcopy(s))
 qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None,validate=False)
rows=[]
for mb in (16,64):
 for seed in SEEDS:
  folder=R/f'batch-{mb}-seed-{seed}';train=torch.load(folder/'iteration-5120.pt',map_location='cpu',weights_only=False);export=torch.load(folder/'inference-5120.pt',map_location='cpu',weights_only=False);assert train['minibatch']==mb and export['channels']==4
  for device in ('cpu','cuda'):
   a=TerminalNet(4).to(device);b=TerminalNet(4).to(device);a.load_state_dict(train['model']);b.load_state_dict(export['model'])
   for state in states:
    obs,qs=observe(state,channels=4);batch=collate([obs],device)
    with torch.no_grad():x=a(*batch);y=b(*batch)
    assert all(torch.equal(u,v) for u,v in zip(x,y))
   rows.append(dict(minibatch=mb,seed=seed,device=device,states=len(states),bit_equal=True))
write(R/'export-audit.json',dict(pass_=True,rows=rows,comparison='training checkpoint vs inference export within each device; not CPU-vs-CUDA bit equality'))
