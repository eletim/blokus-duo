import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from blokus.linear_value_fit import ValueTrainer
from blokus.linear_value import Agent
from blokus.linear_train2 import state_hash
from blokus.linear_rl import prefix
R=Path('artifacts/linear-value-retrain');OLD=Path('artifacts/linear-hybrid/exports');SEEDS=(3101,3102,3103)
def write(n,d):(R/n).write_text(json.dumps(d,indent=2)+'\n')
def exclusions():
 states=set(json.loads(Path('artifacts/linear-value/exclusions.json').read_text())['states']);sources=[]
 for p in Path('artifacts').rglob('*prefixes.json'):
  try:d=json.loads(p.read_text())
  except (ValueError,OSError):continue
  if isinstance(d,dict) and isinstance(d.get('prefixes'),list):
   rows=d['prefixes'];hs=[r['hash'] for r in rows if isinstance(r,dict) and 'hash' in r];states.update(hs)
   if hs:sources.append(dict(path=str(p),count=len(hs),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 with gzip.open(R/'diagnostic.json.gz','rt') as f:d=json.load(f)
 states.update(state_hash(prefix(g['prefix_seed'])) for g in d['games'])
 write('exclusions.json',dict(states=sorted(states),sources=sources));return states

def pilot():
 if (R/'fixed-plan.json').exists():raise FileExistsError('preserve existing plan')
 diag=json.loads((R/'diagnostic-summary.json').read_text());arms=['P','S'] if diag['add_S'] else ['P'];forbidden=exclusions();rows=[];visited=set()
 for arm in arms:
  t=ValueTrainer(9301,Agent.load(OLD/'seed-3101.json'),arm,forbidden=forbidden,series=2330000000);start=time.perf_counter();log,g=t.step(visited);seconds=time.perf_counter()-start
  with gzip.open(R/f'pilot-{arm}.json.gz','wt') as f:json.dump(dict(games=g,log=log),f)
  rows.append(dict(arm=arm,seconds=seconds))
 expected=sum(x['seconds'] for x in rows)*256*3;games=4096 if expected<1500 else 2048
 write('fixed-plan.json',dict(arms=arms,seeds=list(SEEDS),games_per_seed_arm=games,updates=games//16,expected_seconds=expected*games/4096,pilot=rows,diagnostic_seconds=diag['seconds'],cap_seconds=1800,initialization='continue frozen original value, fresh Adam',policy='frozen old greedy',selection='fixed endpoint'))
 write('pilot-visited.json',sorted(visited));print('FIXED',arms,games,expected,flush=True)

def train():
 plan=json.loads((R/'fixed-plan.json').read_text());forbidden=json.loads((R/'exclusions.json').read_text())['states'];visited=set(json.loads((R/'pilot-visited.json').read_text()));start=time.perf_counter();runs=[]
 if list(R.glob('P-*/state.json')) or list(R.glob('S-*/state.json')):raise FileExistsError('preserve trained runs')
 spent=plan['diagnostic_seconds']+sum(x['seconds'] for x in plan['pilot'])
 for seed in plan['seeds']:
  for arm in plan['arms']:
   folder=R/f'{arm}-{seed}';folder.mkdir(exist_ok=True);source=Agent.load(OLD/f'seed-{seed}.json');t=ValueTrainer(seed,source,arm,forbidden=forbidden);logs=[];begin=time.perf_counter()
   with gzip.open(folder/'games.jsonl.gz','wt') as stream:
    while t.iteration<plan['updates']:
     if spent+time.perf_counter()-start>plan['cap_seconds']:t.save(folder/'state.json');raise TimeoutError('finite compute cap')
     log,g=t.step(visited);logs.append(log);stream.write(json.dumps(dict(iteration=t.iteration,games=g))+'\n')
     if t.iteration%64==0:t.save(folder/'state.json');print(arm,seed,t.iteration,flush=True)
   t.save(folder/'state.json');t.agent.save(folder/'policy.json',dict(seed=seed,arm=arm,value_games=plan['games_per_seed_arm'],policy_iteration=1024,source_sha256=hashlib.sha256((OLD/f'seed-{seed}.json').read_bytes()).hexdigest(),policy_frozen=True));write(f'{arm}-{seed}/training.json',dict(logs=logs,seconds=time.perf_counter()-begin));runs.append(dict(seed=seed,arm=arm,seconds=time.perf_counter()-begin,games=plan['games_per_seed_arm']))
 old=np.load('/tmp/linear-value-training.npy');new=np.array(sorted(bytes.fromhex(x) for x in visited),dtype='S32');array=np.unique(np.concatenate([old,new]));np.save('/tmp/linear-retrain-training.npy',array)
 write('training-summary.json',dict(runs=runs,seconds=time.perf_counter()-start,old_training_sha256=hashlib.sha256(old.tobytes()).hexdigest(),new_visited_states=len(new),new_sha256=hashlib.sha256(new.tobytes()).hexdigest(),union_states=len(array),union_sha256=hashlib.sha256(array.tobytes()).hexdigest(),source_history_reference='artifacts/linear-value/training-summary.json'))
 print('TRAIN COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','train']);a=p.parse_args();{'pilot':pilot,'train':train}[a.mode]()
