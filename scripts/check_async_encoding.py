import json,random,time
import torch
import blokus.ppo_frontier as f
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.generation_encoding import encode,child,observe
import frontier_generation as g
torch.set_num_threads(1);rows=[]
for task in json.load(open('artifacts/gpu-generation/bank.json'))['games'][::4]:
 p=PPOAgent(task['seed'],'cpu',backend='native');s=g.ref.e.state(task['prefix']['moves']);r=random.Random(95);a=DeepSearch(p,60000);m=a.act(s,r)
 f.value_inputs=encode;f.child=child;p._observation=lambda s:observe(s,p.channels)
 b=f.FrontierSearch(p,60000);rr=random.Random(95);mm=f.run_frontier([b.act(s,rr)],f.FrontierEvaluator(p))[0]
 assert m==mm and r.getstate()==rr.getstate()
 for k in ('root_scores','nodes','cutoffs','nn_policy_calls','nn_value_calls','cache_hits'):assert a.stats[k]==b.stats[k],k
 rows.append(dict(seed=task['seed'],id=task['id'],nodes=a.stats['nodes'],exact=True))
g.ref.write('artifacts/gpu-async512/singleton-exact.json',rows)
