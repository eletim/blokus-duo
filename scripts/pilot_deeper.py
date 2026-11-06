import sys,json,copy,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from blokus.engine import State,play,legal_moves
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search import SearchAgent
from blokus.ppo_search_deeper import DeepSearch
R=Path('artifacts/gen3-deeper')
torch.set_num_threads(1);p=PPOAgent(1751401,'cpu');rng=random.Random(918171);states=[]
for game in range(3):
 s=State()
 for ply in range(34):
  if s.done:break
  if ply in (4,16,28):states.append(copy.deepcopy(s))
  qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=False)
rows=[]
for budget in (500,2000):
 for repeat in (0,1):
  for i,s in enumerate(states):
   for name in (('E','H') if repeat==0 else ('H','E')):
    a=SearchAgent(p,budget) if name=='E' else DeepSearch(p,budget);m=a.act(s,random.Random(9));rows.append(dict(budget=budget,repeat=repeat,state=i,method=name,**a.stats))
(R/'pilot.json').write_text(json.dumps({'rows':rows,'states':[s.__dict__ for s in states]},indent=2))
for budget in (500,2000):
 for name in ('E','H'):
  a=[x for x in rows if x['budget']==budget and x['method']==name];print(budget,name,'mean',sum(x['elapsed_ms'] for x in a)/len(a),'depth6',sum(x['depth']==6 for x in a),'n',len(a),flush=True)
