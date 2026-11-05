import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.ppo_6144_eval import prepare
from blokus import ppo_6144_eval as old
from blokus.research import sha
from blokus.value_experiment import write
R=Path('artifacts/ppo-generation-pool');old.R=R
prepare()
d=json.loads((R/'lock.json').read_text());d.pop('latest',None);d.pop('criterion_weights',None);d['previous']=json.loads(Path('artifacts/ppo-6144/selection.json').read_text());d['resume']={};fixed={}
for seed in (1401,1402,1403):
 k=str(seed);p=Path(f'artifacts/ppo-6144/seed-{seed}/iteration-6144.pt');d['resume'][k]=dict(path=str(p),sha256=sha(p));g=d['gen2'][k];p=Path(f'artifacts/ppo-learning-curve/seed-{seed}/iteration-2048.pt');assert p.exists()
 fixed[k]={'gen2':dict(path=g['path'],sha256=g['sha256']),'gen2048':dict(path=str(p),sha256=sha(p))}
d.update(fixed=fixed,validation_seed=1670000,holdout_seed=1680000,common_max_iteration=7168)
write(R/'lock.json',d)
for kind in ('A','B'):write(R/f'pool-{kind}.json',dict(kind=kind,fixed=fixed if kind=='B' else {},schedule='(iteration+past_slot)%3; past,gen2,gen2048',game_ratios={'current':.5,'heuristic':.25,**({'past':.25} if kind=='A' else {'past':1/12,'gen2':1/12,'gen2048':1/12})}))
write(R/'started.json',dict(time=time.time()))
