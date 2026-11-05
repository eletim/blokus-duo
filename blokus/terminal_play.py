"""CLI for the research terminal PPO policy, without search."""
import argparse,random,json
from .engine import State,Move,play
from .terminal_eval import agent
from .ai import seed_all

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--opponent',default='heuristic');p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--seed',type=int,default=1);a=p.parse_args();seed_all(a.seed);left=agent(a.model);right=agent(a.opponent);s=State();rng=random.Random(a.seed);moves=[]
 while not s.done:
  if s.turn==a.human_seat:
   print(s.render());raw=input('piece orientation row col / pass: ').strip()
   try:m=None if raw=='pass' else Move(*map(int,raw.split()));play(s,m)
   except (ValueError,TypeError):print('合法手を入力してください');continue
  else:m=(left if s.turn==0 else right).act(s,rng);play(s,m)
  moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
 print(json.dumps(dict(scores=[s.score(0),s.score(1)],moves=moves)))
if __name__=='__main__':main()
