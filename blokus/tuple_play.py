"""Portable CPU terminal-WDL tuple/linear value with fixed-width non-neural search."""
import argparse,json,random
from .engine import State,play,legal_moves
from .tuple_value import load_value
from .linear_tree import CompletedSearch
from .linear_value import Agent

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--backend',choices=['native','python'],default='python');p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--seed',type=int,default=1);a=p.parse_args()
 bot=CompletedSearch(load_value(a.checkpoint,a.backend),ms=100,width=24,replies=8);other=Agent('old',backend=a.backend);s=State();rng=random.Random(a.seed);moves=[]
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
