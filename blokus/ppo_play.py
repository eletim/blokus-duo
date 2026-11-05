"""PPO vs human/AI, portable CPU/CUDA inference. Existing defaults unchanged."""
import argparse,json,random
from pathlib import Path
from .engine import State,legal_moves,play
from .ai import select,seed_all
from .ppo_policy import PPOAgent

class HeuristicAgent:
 def act(self,s,rng):return select(s,'heuristic',rng)

def human_move(s,input_fn=input,output=print):
 qs=legal_moves(s)
 if not qs:output('合法手なし: pass');return None
 for i,q in enumerate(qs):output(f'{i}: {q.move}')
 while True:
  try:
   ix=int(input_fn('合法手番号: '))
   if 0<=ix<len(qs):return qs[ix].move
  except ValueError:pass
  output('表示された合法手番号を入力してください')

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-seed',type=int,choices=[1401,1402,1403],default=1401);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--opponent',choices=['heuristic','ppo','oldvalue','teacher100'],default='heuristic');p.add_argument('--opponent-model-seed',type=int,choices=[1401,1402,1403],default=1402);p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--seed',type=int,default=1);a=p.parse_args();seed_all(a.seed)
 bot=PPOAgent(a.model_seed,a.device);base=Path(__file__).resolve().parents[1]/'artifacts/value-search/compact/seed-201/predictive-best.pt'
 if a.human_seat is not None:other=bot
 elif a.opponent=='ppo':other=PPOAgent(a.opponent_model_seed,a.device)
 elif a.opponent=='oldvalue':
  from .value_search import BudgetAgent
  other=BudgetAgent('value',base,.5,24,6,bot.device)
 elif a.opponent=='teacher100':
  from .refinement_search import RefinementAgent
  other=RefinementAgent(base,100,'priority',bot.device)
 else:other=HeuristicAgent()
 bots=[bot,other];s=State();rng=random.Random(a.seed);moves=[]
 try:
  while not s.done:
   if s.turn==a.human_seat:print(s.render());m=human_move(s)
   else:m=bots[s.turn].act(s,rng)
   play(s,m,validate=True);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
 except (EOFError,KeyboardInterrupt):print('\n対局を終了しました');return
 print(json.dumps(dict(scores=[s.score(0),s.score(1)],moves=moves,model_seed=a.model_seed,device=bot.device)))
if __name__=='__main__':main()
