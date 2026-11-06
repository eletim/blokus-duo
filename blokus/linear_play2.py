"""CPU-only non-neural policy/value inference, optional bounded minimax."""
import argparse,json,random
from .linear_value import Agent,Search
from .linear_tree import CompletedSearch
from .engine import State,play

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--opponent-checkpoint');p.add_argument('--value-checkpoint',help='explicit v2 value source; policy weights remain from --checkpoint');p.add_argument('--backend',choices=['python','native'],default='python');p.add_argument('--search-ms',type=float,default=0);p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--seed',type=int,default=1);p.add_argument('--completed-search',action='store_true');p.add_argument('--root-width',type=int,default=6);p.add_argument('--reply-width',type=int,default=3);p.add_argument('--depth',type=int,choices=[2,4],default=2);a=p.parse_args()
 model=Agent.load_pair(a.checkpoint,a.value_checkpoint,a.backend) if a.value_checkpoint else Agent.load(a.checkpoint,a.backend);bot=(CompletedSearch(model,ms=a.search_ms,width=a.root_width,replies=a.reply_width,depth=a.depth) if a.completed_search else Search(model,ms=a.search_ms)) if a.search_ms else model;other=Agent.load(a.opponent_checkpoint,a.backend) if a.opponent_checkpoint else Agent('old',backend=a.backend);s=State();rng=random.Random(a.seed);moves=[]
 try:
  while not s.done:
   if s.turn==a.human_seat:
    from .ppo_play import human_move
    print(s.render());m=human_move(s)
   else:m=(bot if a.human_seat is not None or s.turn==0 else other).act(s,rng)
   play(s,m);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
 except (EOFError,KeyboardInterrupt):print('対局を終了しました');return
 print(json.dumps(dict(scores=[s.score(0),s.score(1)],moves=moves,search_ms=a.search_ms)))
if __name__=='__main__':main()
