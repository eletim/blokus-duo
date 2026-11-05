"""Experimental fixed-PPO short search CLI; global defaults remain unchanged."""
import argparse,json,random
from .ppo_short_search import ShortSearch
from .terminal_ppo import TerminalAgent
from .terminal_eval import Heuristic
from .engine import State,Move,play
from .ai import seed_all

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='artifacts/ppo-gen2/seed-1401.pt');p.add_argument('--budget-ms',type=int,choices=[6,25],default=25);p.add_argument('--margin',type=float,choices=[0.,.15],default=0.);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--opponent',choices=['self','heuristic'],default='self');p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--seed',type=int,default=1);a=p.parse_args();seed_all(a.seed)
 search=ShortSearch(a.model,a.budget_ms,a.margin,a.device);other=TerminalAgent(a.model,a.device) if a.opponent=='self' else Heuristic();s=State();rng=random.Random(a.seed);moves=[]
 while not s.done:
  if s.turn==a.human_seat:
   print(s.render())
   try:
    raw=input('piece orientation row col / pass: ').strip();m=None if raw=='pass' else Move(*map(int,raw.split()));play(s,m)
   except (ValueError,TypeError):print('合法手を入力してください');continue
  else:
   ag=search if a.human_seat is not None or s.turn==0 else other;m=ag.act(s,rng);play(s,m)
  moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
 print(json.dumps(dict(scores=[s.score(0),s.score(1)],moves=moves,search_calls=len(search.records))))
if __name__=='__main__':main()
