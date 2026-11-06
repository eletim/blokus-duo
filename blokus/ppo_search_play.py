"""Research CLI; fixed gen3, experimental old/batched/exact search, no model changes."""
import argparse,json,random
import torch
from .engine import State,play
from .ppo_policy import PPOAgent
from .ppo_play import human_move
from .ppo_search import SearchAgent
from .ppo_search_batch import BatchSearch

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['old','B','C'],default='B');p.add_argument('--budget-ms',type=int,choices=[500,2000],default=500);p.add_argument('--model-seed',type=int,choices=[1751401,1751402,1751403],default=1751401);p.add_argument('--human-seat',type=int,choices=[0,1]);a=p.parse_args();torch.set_num_threads(1)
 policy=PPOAgent(a.model_seed,'cpu',generation='gen3',backend='python');bot=SearchAgent(policy,a.budget_ms) if a.method=='old' else BatchSearch(policy,a.budget_ms,exact=a.method=='C');s=State();rng=random.Random(1);moves=[];stats=[]
 try:
  while not s.done:
   if s.turn==a.human_seat:print(s.render());m=human_move(s)
   elif a.human_seat is not None or s.turn==0:m=bot.act(s,rng);stats.append(bot.stats)
   else:m=policy.act(s,rng)
   play(s,m,validate=True);moves.append(None if m is None else list(m.__dict__.values()))
 except (EOFError,KeyboardInterrupt):print('対局を終了しました');return
 print(json.dumps(dict(method=a.method,budget_ms=a.budget_ms,model_seed=a.model_seed,scores=[s.score(0),s.score(1)],moves=moves,search_stats=stats)))
if __name__=='__main__':main()
