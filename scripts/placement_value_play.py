"""History-free research value export + unchanged Frozen gen3 policy, CPU CLI."""
import argparse,hashlib,json,random
from pathlib import Path
import torch
from blokus.engine import State,play
from blokus.ppo_policy import PPOAgent
from blokus.ppo_play import human_move,HeuristicAgent
from blokus.placement_value import PlacementValue,PlacementEvaluator
from blokus.ppo_search_deeper import DeepSearch

def load(path,backend='python',budget=500,depth=6):
 torch.set_num_threads(1);ck=torch.load(path,map_location='cpu',weights_only=True)
 if ck.get('format')!='placement-value-v1':raise ValueError('not a placement-value-v1 inference export')
 seed=ck['seed'];source=Path(__file__).resolve().parents[1]/f'artifacts/ppo-gen3/seed-{seed}.pt'
 if hashlib.sha256(source.read_bytes()).hexdigest()!=ck['source_sha']:raise ValueError('Frozen gen3 source hash mismatch')
 p=PPOAgent(seed,'cpu',generation='gen3',backend=backend);m=PlacementValue(p.model,ck['variant']);m.load_state_dict(ck['model']);return DeepSearch(p,budget,max_depth=depth,value_evaluator=PlacementEvaluator(m,backend=backend)),ck

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--backend',choices=['python','native'],default='python');p.add_argument('--budget-ms',type=float,default=500);p.add_argument('--max-depth',type=int,choices=[2,4,6],default=6);p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--seed',type=int,default=1);a=p.parse_args();bot,ck=load(a.checkpoint,a.backend,a.budget_ms,a.max_depth);other=HeuristicAgent();s=State();rng=random.Random(a.seed);moves=[];stats=[]
 try:
  while not s.done:
   if s.turn==a.human_seat:print(s.render());m=human_move(s)
   elif a.human_seat is not None or s.turn==0:m=bot.act(s,rng);stats.append(dict(bot.stats))
   else:m=other.act(s,rng)
   play(s,m,validate=True);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
 except (EOFError,KeyboardInterrupt):print('対局を終了しました');return
 print(json.dumps(dict(scores=[s.score(0),s.score(1)],moves=moves,stats=stats,model_seed=ck['seed'],variant=ck['variant'],checkpoint_sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),epochs=ck.get('epochs'),device='cpu')))
if __name__=='__main__':main()
