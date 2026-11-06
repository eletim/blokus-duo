import json
from pathlib import Path
import torch
from blokus.ppo_policy import TerminalNet,PPOAgent,observe,collate
from blokus.engine import State,Move,play
R=Path('artifacts/ppo-four-gpu');sel=json.loads((R/'selection.json').read_text());results=[]
for seed,v in sel['selected'].items():
 source=R/f'seed-{seed}'/f"iteration-{v['iteration']}.pt";ck=torch.load(source,map_location='cpu',weights_only=False)
 for device in ('cpu','cuda'):
  m=TerminalNet(4).to(device).eval();m.load_state_dict(ck['model']);agent=PPOAgent(checkpoint=f'artifacts/ppo-four-gpu-models/seed-{seed}.pt',device=device,backend='native');batch=collate([observe(State(),channels=4)[0]],device)
  with torch.no_grad():a=m(*batch);b=agent.model(*batch)
  assert all(torch.equal(x,y) for x,y in zip(a,b));results.append(dict(seed=seed,device=device,logits_value_bit_equal=True))
for mode in ('cpu','cuda'):
 game=json.loads((R/f'cli-{mode}.json').read_text());s=State()
 for row in game['moves']:
  m=Move(*row) if row else None
  from blokus.engine import is_legal,legal_moves
  assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False)
 assert s.done and game['scores']==[s.score(0),s.score(1)]
(R/'export-cli-audit.json').write_text(json.dumps(dict(pass_=True,outputs=results,cli_games_replayed=True),indent=2));print('EXPORT CLI PASS')
