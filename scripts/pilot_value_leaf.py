import sys,copy,random,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from blokus.engine import State,play
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search import SearchAgent
from blokus.ppo_search_staged import ValueLeafSearch
R=Path('artifacts/gen3-value-leaf');torch.set_num_threads(1);states=[State(*x) for x in json.loads((R/'equivalence.json').read_text())['states']];rows=[]
for backend in ('python','native'):
 p=PPOAgent(1751401,'cpu',backend=backend);p.act(State(),random.Random(1))
 for ms in (500,2000):
  for rep in range(2):
   for i in (0,3,6,9,24,27,30,33):
    methods=['A','E','D'] if rep==0 else ['D','E','A']
    for method in methods:
     a=SearchAgent(p,ms) if method=='A' else ValueLeafSearch(p,ms,root_stages=(8,12) if method=='D' else (),reuse=method=='D')
     m=a.act(states[i],random.Random(77));stats=dict(a.stats);stats.pop('root_scores',None)
     rows.append(dict(backend=backend,budget_ms=ms,repeat=rep,state=i,method=method,move=None if m is None else list(m.__dict__.values()),**stats))
(R/'pilot.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n');print('pilot rows',len(rows))
