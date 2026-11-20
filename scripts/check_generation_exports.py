"""CPU-only restoration of unchanged public gen3 and existing3072A exports."""
import hashlib,json,random
from pathlib import Path
import torch
from blokus.ppo_policy import PPOAgent
from blokus.engine import State,play,is_legal
from blokus.placement_value import PlacementValue,PlacementEvaluator
from blokus.ppo_search_deeper import DeepSearch
torch.set_num_threads(1);rows=[]
for seed in (1751401,1751402,1751403):
 path=Path(f'artifacts/ppo-gen3/seed-{seed}.pt');source=hashlib.sha256(path.read_bytes()).hexdigest();p=PPOAgent(seed,'cpu',backend='python');export=Path(f'artifacts/value-placement-scale/n3072/A-{seed}.pt');ck=torch.load(export,map_location='cpu',weights_only=True);assert ck['source_sha']==source
 m=PlacementValue(p.model,ck['variant']);m.load_state_dict(ck['model']);ev=PlacementEvaluator(m,backend='python');s=State();rng=random.Random(82)
 for _ in range(12):play(s,p.act(s,rng),validate=True)
 v=ev(s);assert -1<=v<=1;a=DeepSearch(p,30000,max_depth=2,value_evaluator=ev);move=a.act(s,rng);assert is_legal(s,move) and a.stats['depth']==2
 rows.append(dict(seed=seed,source_sha=source,export_sha=hashlib.sha256(export.read_bytes()).hexdigest(),value=v,move=None if move is None else [move.piece,move.orientation,move.row,move.col],depth=a.stats['depth']))
print(json.dumps(rows,indent=2))
