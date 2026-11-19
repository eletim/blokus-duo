"""CPU-only-device inference in a history-free source archive; no training."""
import json,hashlib,random,sys
from pathlib import Path
import torch
from blokus.engine import State,play
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.placement_value import PlacementValue,PlacementEvaluator

def main():
 torch.set_num_threads(1);rows=[]
 for seed in (1751401,1751402,1751403):
  path=Path(f'artifacts/ppo-gen3/seed-{seed}.pt');sha=hashlib.sha256(path.read_bytes()).hexdigest()
  p=PPOAgent(seed,'cpu',backend='python');s=State();rng=random.Random(11)
  moves=[]
  while not s.done:
   m=p.act(s,rng);play(s,m,validate=True);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
  source=Path(f'artifacts/value-placement-scale/n3072/A-{seed}.pt')
  ck=torch.load(source,map_location='cpu',weights_only=True);assert ck['source_sha']==sha
  model=PlacementValue(p.model,'A');model.load_state_dict(ck['model'])
  value=PlacementEvaluator(model,backend='python');z=State();v=value(z)
  search=DeepSearch(p,60000,max_depth=2,value_evaluator=value);search.act(z,random.Random(5));assert search.stats['depth']==2
  rows.append(dict(seed=seed,source_sha=sha,placement_sha=hashlib.sha256(source.read_bytes()).hexdigest(),value=v,complete_cpu_game=len(moves),scores=[s.score(0),s.score(1)],depth=search.stats['depth']))
 print(json.dumps(dict(cuda_visible=torch.cuda.is_available(),torch_version=torch.__version__,device='cpu',history_required=False,rows=rows),indent=2))
if __name__=='__main__':main()
