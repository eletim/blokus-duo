"""CPU-only research policy + unchanged Frozen gen4 value human/AI play."""
import argparse,json,random
from pathlib import Path
import torch
from blokus.gen4 import load,verify
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.ppo_play import human_move,HeuristicAgent
from blokus.engine import State,play
import placement_experiment as e

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,required=True);p.add_argument('--seed',type=int,default=1751401,choices=[1751401,1751402,1751403]);p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--backend',choices=['native','python'],default='native');a=p.parse_args()
 torch.set_num_threads(1);verify();ck=torch.load(a.policy,map_location='cpu',weights_only=True);meta=ck['candidate_policy'];assert meta['seed']==a.seed and meta['gen4_value_sha']==e.sha(f'models/gen4/value-{a.seed}.pt')
 _,value=load(a.seed,a.backend);policy=PPOAgent(a.seed,'cpu',checkpoint=a.policy,backend=a.backend);bot=DeepSearch(policy,500,width=4,max_depth=6,value_evaluator=value);other=HeuristicAgent();s=State();rng=random.Random(1);moves=[]
 try:
  while not s.done:
   if s.turn==a.human_seat:print(s.render());m=human_move(s)
   elif a.human_seat is not None or s.turn==0:m=bot.act(s,rng)
   else:m=other.act(s,rng)
   play(s,m,validate=True);moves.append(e.raw(m))
 except (EOFError,KeyboardInterrupt):print('対局を終了しました');return
 print(json.dumps(dict(mode='research-candidate-policy-frozen-gen4-value',policy_sha=e.sha(a.policy),metadata=meta,moves=moves,scores=[s.score(0),s.score(1)])))
if __name__=='__main__':main()
