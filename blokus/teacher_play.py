"""Optional validated 100ms teacher; existing low-budget defaults unchanged."""
import argparse,random,json
from .ai import seed_all
from .engine import State,play,legal_moves
from .refinement_search import RefinementAgent
from .value_search import BudgetAgent

def main():
 p=argparse.ArgumentParser();p.add_argument('--human',action='store_true');p.add_argument('--seed',type=int,default=100);p.add_argument('--device',default='cuda');a=p.parse_args();seed_all(a.seed)
 base='artifacts/value-search/compact/seed-201/predictive-best.pt'
 bots=[RefinementAgent(base,100,'priority',a.device),BudgetAgent('value',base,.5,24,100,a.device)]
 s=State();rng=random.Random(a.seed)
 while not s.done:
  if a.human and s.turn==1:
   print(s.render());qs=legal_moves(s)
   if qs:
    for i,q in enumerate(qs):print(i,q.move)
    while True:
     try:m=qs[int(input('合法手番号: '))].move;break
     except (ValueError,IndexError):print('番号を確認してください')
   else:m=None
  else:m=bots[s.turn].act(s,rng)
  play(s,m,validate=True)
 print(json.dumps(dict(scores=[s.score(0),s.score(1)],teacher=bots[0].stats(),old=bots[1].stats())))
if __name__=='__main__':main()
