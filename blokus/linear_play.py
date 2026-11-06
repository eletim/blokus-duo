"""Play the experimental non-neural policy through the existing terminal UI."""
import argparse,json,random
from .engine import State,play
from .linear_rl import LinearAgent


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--backend',choices=['python','native'],default='python');p.add_argument('--seed',type=int,default=1);a=p.parse_args()
 bot=LinearAgent.load(a.checkpoint,a.backend);other=LinearAgent(backend=a.backend)
 s=State();rng=random.Random(a.seed);moves=[]
 try:
  while not s.done:
   if s.turn==a.human_seat:
    from .ppo_play import human_move
    print(s.render());m=human_move(s)
   else:m=(bot if a.human_seat is not None or s.turn==0 else other).act(s,rng)
   play(s,m,validate=True);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
 except (EOFError,KeyboardInterrupt):print('\n対局を終了しました');return
 print(json.dumps(dict(scores=[s.score(0),s.score(1)],moves=moves)))
if __name__=='__main__':main()
