"""Same-process suspended-game GPU frontier, fixed-work benchmark and restart."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,gzip,json,random,time
from pathlib import Path
import numpy as np
import torch
import gpu_generation as ref
from blokus.ppo_frontier import FrontierSearch,FrontierEvaluator,run_frontier
from blokus.ppo_policy import PPOAgent
from blokus.engine import play
R=Path('artifacts/gpu-frontier')

def prepare():
 assert not (R/'bank.json').exists()
 ex=set(json.loads((ref.R/'excluded.json').read_text())['states'])
 from blokus.ppo_runtime import state_hash
 for p in (ref.R/'cpu-1').glob('*.json.gz'):
  d=json.load(gzip.open(p,'rt'));s=ref.e.state([])
  from blokus.engine import Move
  for m in d['moves']:
   ex.add(state_hash(s));play(s,Move(*m) if m else None)
 idx=np.load(ref.e.INDEX,mmap_mode='r');games=[]
 for seed in ref.e.SEEDS:
  for gid in range(64):
   rs=723000000+seed*1000+gid;rng=random.Random(rs);pr=ref.e.prefix(rng,idx,ex,8);ex.add(pr['state_hash'])
   games.append(dict(seed=seed,id=gid,rng_seed=rs,prefix=pr,rng_state=rng.getstate()))
 ref.write(R/'bank.json',dict(games=games,models={str(s):ref.e.sha(f'artifacts/ppo-gen3/seed-{s}.pt') for s in ref.e.SEEDS}))

def game(task,policy,directory):
 path=directory/f"{task['seed']}-{task['id']}.json.gz"
 if path.exists():
  old=json.load(gzip.open(path,'rt'));assert old['task']==task;return old
 rng=random.Random();rng.setstate(ref.tuples(task['rng_state']));s=ref.e.state(task['prefix']['moves']);moves=list(task['prefix']['moves']);stats=[]
 while not s.done:
  a=FrontierSearch(policy,3600000);m=yield from a.act(s,rng)
  assert a.stats['forced_pass'] or a.stats['depth']==6
  stats.append(copy.deepcopy(a.stats));play(s,m,validate=True);moves.append(ref.e.raw(m))
 out=dict(task=task,moves=moves,scores=[s.score(0),s.score(1)],stats=stats)
 tmp=path.with_suffix('.tmp')
 with gzip.open(tmp,'wt') as f:json.dump(out,f)
 tmp.replace(path);return out

def bench(repeat,device='cuda',bank_path=None):
 bank_path=Path(bank_path or R/'bank.json');bank=json.loads(bank_path.read_text());directory=R/f'frontier-{repeat}';directory.mkdir(exist_ok=True)
 output=R/f'frontier-{repeat}.json';assert not output.exists()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 begin=time.perf_counter();agents={s:PPOAgent(s,device,backend='native') for s in ref.e.SEEDS}
 assert all(ref.e.sha(f'artifacts/ppo-gen3/seed-{s}.pt')==h for s,h in bank['models'].items())
 init=time.perf_counter()-begin;start=time.perf_counter();cpu=time.process_time();groups=[]
 if device=='cuda':torch.cuda.reset_peak_memory_stats()
 for seed in ref.e.SEEDS:
  tasks=[x for x in bank['games'] if x['seed']==seed];ev=FrontierEvaluator(agents[seed]);t=time.perf_counter()
  rows=run_frontier([game(g,agents[seed],directory) for g in tasks],ev)
  groups.append(dict(seed=seed,seconds=time.perf_counter()-t,games=len(rows),batches=ev.hist,pack_seconds=ev.pack_seconds,forward_seconds=ev.forward_seconds,return_seconds=ev.return_seconds))
 out=dict(seconds=time.perf_counter()-start,initialization_seconds=init,cpu_seconds=time.process_time()-cpu,groups=groups,bank_sha=ref.e.sha(bank_path),games=len(bank['games']),device=device,peak_memory=torch.cuda.max_memory_allocated() if device=='cuda' else None)
 ref.write(output,out);print(json.dumps(out),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','cpu','gpu']);p.add_argument('--repeat',type=int,default=1);p.add_argument('--bank');a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='cpu':ref.R=R;ref.benchmark('cpu',a.repeat)
 else:bench(a.repeat,bank_path=a.bank)
