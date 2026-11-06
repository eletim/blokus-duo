import gzip,json
from blokus.ppo_lr64 import R,write
from blokus.ppo_gpu_train import sha
rows=[]
for c in 'ABC':
 p=R/f'pilot-{c}-full/history-4097-4104.jsonl.gz'
 with gzip.open(p,'rt') as f:rows.append(json.loads(next(f)))
assert rows[0]==rows[1]==rows[2]
write(R/'initial-data-audit.json',dict(pass_=True,identical_initial_16_games_and_targets=True,series_seed=1771401,base='artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt',base_sha256=sha('artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt'),note='Same restored RNG/model/past; trajectories can diverge only after condition-specific updates.'))
