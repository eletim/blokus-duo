"""Research occupancy value CPU CLI; leaves public gen4 unchanged."""
from future_common import *
from future_eval import model
from blokus.ppo_play import human_move,HeuristicAgent
import argparse
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=list('ABC'),default='C');p.add_argument('--seed',type=int,default=1751401);p.add_argument('--human-seat',type=int,choices=[0,1]);a=p.parse_args();torch.set_num_threads(1);pol,val=model(a.seed,a.arm);bot=DeepSearch(pol,500,width=4,max_depth=6,value_evaluator=val);other=HeuristicAgent();s=State();rng=random.Random(924);moves=[]
 while not s.done:
  if s.turn==a.human_seat:print(s.render());m=human_move(s)
  elif a.human_seat is not None or s.turn==0:m=bot.act(s,rng)
  else:m=other.act(s,rng)
  play(s,m,validate=True);moves.append(raw(m))
 print(json.dumps(dict(mode='future-occupancy-research',arm=a.arm,seed=a.seed,moves=moves,scores=[s.score(0),s.score(1)])))
