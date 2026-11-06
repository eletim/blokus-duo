"""CPU-only non-neural linear policy/value candidate; no training history required."""
import argparse,json,random
from pathlib import Path
from .engine import State,play,legal_moves
from .linear_value import Agent
from .linear_tree import CompletedSearch

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-seed',type=int,choices=[3101,3102,3103],default=3101);p.add_argument('--checkpoint');p.add_argument('--backend',choices=['python','native'],default='python');p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--random-seed',type=int,default=1);a=p.parse_args()
 path=Path(a.checkpoint) if a.checkpoint else Path(__file__).resolve().parents[1]/'models'/'linear-L'/f'seed-{a.model_seed}.json'
 bot=CompletedSearch(Agent.load(path,a.backend),ms=100,width=24,replies=8,depth=2);other=Agent('old',backend=a.backend);s=State();rng=random.Random(a.random_seed);moves=[]
 try:
  while not s.done:
   if s.turn==a.human_seat:
    qs=legal_moves(s)
    if not qs:m=None
    else:
     print(s.render())
     for i,q in enumerate(qs):print(i,q.move)
     while True:
      try:
       i=int(input('合法手番号: '))
       if 0<=i<len(qs):m=qs[i].move;break
      except ValueError:pass
      print('表示された合法手番号を入力してください')
   else:m=(bot if a.human_seat is not None or s.turn==0 else other).act(s,rng)
   play(s,m);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
 except (EOFError,KeyboardInterrupt):print('対局を終了しました');return
 print('\n'+json.dumps(dict(scores=[s.score(0),s.score(1)],moves=moves)))
if __name__=='__main__':main()
