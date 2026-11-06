import argparse,json,time,gzip,hashlib
from pathlib import Path
import numpy as np
from blokus.linear_train2 import Trainer,state_hash
from blokus.linear_rl import prefix
R=Path('artifacts/linear-value')
def write(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
def exclusions():
 states=set();sources=[]
 for root in [Path('/home/eletim/blokus-duo-ai/artifacts'),Path('artifacts')]:
  for p in root.rglob('*prefix*.json'):
   try:
    d=json.loads(p.read_text())
    for row in d.get('prefixes',[]):
     if 'moves' in row:
      from blokus.engine import State,Move,play
      s=State()
      for a in row['moves']:play(s,Move(*a) if a else None)
      states.add(state_hash(s))
    sources.append(str(p))
   except (ValueError,KeyError,TypeError,AttributeError):continue
 p=Path('artifacts/linear-rl-pilot/evaluation.json')
 for seed in {x['prefix_seed'] for x in json.loads(p.read_text())}:states.add(state_hash(prefix(seed)))
 # Prior registry also covers older branches whose banks are not in this checkout.
 p=Path('/home/eletim/blokus-duo-ai/artifacts/ppo-confirmation/excluded-starts.json')
 states.update(json.loads(p.read_text())['states']);write('exclusions.json',dict(states=sorted(states),sources=sources,count=len(states)));return states

def pilot():
 forbidden=exclusions();rows=[]
 for mode in ('old','new'):
  t=Trainer(9101,mode,forbidden=forbidden);start=time.perf_counter()
  for _ in range(32):row,g=t.step()
  rows.append(dict(mode=mode,iterations=32,games=512,seconds=time.perf_counter()-start,last=row))
 expected=sum(x['seconds']*1024/32*3 for x in rows);end=1024 if expected<1800 else 512
 write('fixed-plan.json',dict(pilot=rows,expected_training_seconds=expected,iterations=end,seeds=[3101,3102,3103],modes=['old','new'],total_games=end*16*6,training_cap_seconds=5400,selection='fixed final endpoint; no validation-based checkpoint selection',created_before_training=True));print(rows,end,flush=True)

def train():
 if any((R/f'{mode}-{seed}/state.json').exists() for mode in ('old','new') for seed in (3101,3102,3103)):raise FileExistsError('preserve completed experiments; choose --output NEW_DIR and run pilot first')
 plan=json.loads((R/'fixed-plan.json').read_text());forbidden=json.loads((R/'exclusions.json').read_text())['states'];visited=set();begin=time.perf_counter();reports=[]
 for seed in plan['seeds']:
  for mode in plan['modes']:
   folder=R/f'{mode}-{seed}';folder.mkdir(exist_ok=True);t=Trainer(seed,mode,forbidden=forbidden);t.agent.save(folder/'initial.json');logs=[];start=time.perf_counter()
   with gzip.open(folder/'games.jsonl.gz','wt') as stream:
    while t.iteration<plan['iterations']:
     if time.perf_counter()-begin>plan['training_cap_seconds']:raise TimeoutError('finite training cap exceeded; retain partial runs')
     row,games=t.step(visited);logs.append(row);stream.write(json.dumps(dict(iteration=t.iteration,games=games))+'\n')
     if t.iteration in (256,512,1024):t.agent.save(folder/f'policy-{t.iteration}.json',dict(seed=seed,iteration=t.iteration));t.save(folder/'state.json')
     if t.iteration%128==0:print(mode,seed,t.iteration,round(time.perf_counter()-start,1),flush=True)
   t.save(folder/'state.json');t.agent.save(folder/'policy.json',dict(seed=seed,iteration=t.iteration));seconds=time.perf_counter()-start;write(f'{mode}-{seed}/training.json',dict(seconds=seconds,logs=logs));reports.append(dict(mode=mode,seed=seed,seconds=seconds,games=t.iteration*16))
 array=np.array(sorted(bytes.fromhex(x) for x in visited),dtype='S32');np.save('/tmp/linear-value-training.npy',array)
 with gzip.open(R/'training-states.bin.gz','wb') as f:f.write(array.tobytes())
 write('training-summary.json',dict(runs=reports,seconds=time.perf_counter()-begin,unique_states=len(array),state_sha256=hashlib.sha256(array.tobytes()).hexdigest()));print('TRAIN COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','train']);p.add_argument('--output',default=str(R));a=p.parse_args();R=Path(a.output);R.mkdir(parents=True,exist_ok=True);{'pilot':pilot,'train':train}[a.mode]()
