"""Research CPU inference only, frozen-value distilled policy; never changes public defaults."""
import argparse,random,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from blokus.engine import State,play
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.ppo_play import human_move
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--model-seed',type=int,choices=[1751401,1751402,1751403],default=1751401);p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--search',action='store_true');a=p.parse_args();torch.set_num_threads(1)
policy=PPOAgent(a.model_seed,'cpu',checkpoint=a.checkpoint);bot=DeepSearch(policy,500) if a.search else policy;s=State();rng=random.Random(1)
try:
 while not s.done:
  if s.turn==a.human_seat:print(s.render());m=human_move(s)
  else:m=bot.act(s,rng)
  play(s,m,validate=True)
 print(json.dumps(dict(scores=[s.score(0),s.score(1)],research=True,checkpoint=a.checkpoint)))
except (EOFError,KeyboardInterrupt):print('対局を終了しました')
