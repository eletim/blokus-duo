import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import cProfile,pstats,random,json
from pathlib import Path
import torch
import gpu_generation as g
from blokus.ppo_policy import PPOAgent
from blokus.ppo_frontier import FrontierSearch,FrontierEvaluator,run_frontier
p=PPOAgent(1751401,'cuda',backend='native');torch.set_num_threads(1)
bank=json.loads((g.R/'bank.json').read_text());ss=[g.e.state(x['prefix']['moves']) for x in bank['games'][:8]]
pr=cProfile.Profile();pr.enable();ev=FrontierEvaluator(p)
run_frontier([FrontierSearch(p,3600000).act(s,random.Random(i)) for i,s in enumerate(ss)],ev)
pr.disable()
with open('artifacts/gpu-frontier/profile.txt','w') as f:pstats.Stats(pr,stream=f).strip_dirs().sort_stats('cumtime').print_stats(45)
print(vars(ev)|{'policy':'omitted'})
