"""Fixed-tree generation, identical L/T data and finite NumPy learning."""
import argparse,copy,gzip,hashlib,json,random,time,os
from pathlib import Path
import numpy as np
from blokus.engine import State,Move,play
from blokus.linear_rl import prefix
from blokus.linear_value import Agent,value_features,outcome
from blokus.linear_train2 import state_hash
from blokus.linear_tree import CompletedSearch
from blokus.tuple_value import codes,phase,Learner,geometry,SIZE,residual
from scripts.linear_value_eval import sha,contains
R=Path('artifacts/linear-tuple');SEEDS=(3101,3102,3103)
def write(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
def source(seed):return Path(f'artifacts/linear-hybrid/exports/seed-{seed}.json')
def indices():return [np.load(p,mmap_mode='r') for p in ['/tmp/linear-retrain-training.npy','/tmp/ppo-lr64-training.npy','/tmp/ppo-minibatch-training.npy','/tmp/ppo-four-gpu-all-training.npy']]
def prepare():
 old=json.loads(Path('artifacts/linear-search/exclusions.json').read_text());forbidden=set(old['states'])
 for p in Path('artifacts/linear-search').glob('*-prefixes.json'):forbidden.update(x['hash'] for x in json.loads(p.read_text())['prefixes'])
 write('exclusions.json',dict(states=sorted(forbidden),parent_sha=sha('artifacts/linear-search/exclusions.json')))
 write('models.json',[dict(path=str(source(s)),sha256=sha(source(s))) for s in SEEDS]+[dict(path=f'artifacts/ppo-gen3/seed-{1751401+s-3101}.pt',sha256=sha(f'artifacts/ppo-gen3/seed-{1751401+s-3101}.pt')) for s in SEEDS])
 write('indices.json',[dict(path=str(p),states=len(a),sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest()) for p,a in zip(['/tmp/linear-retrain-training.npy','/tmp/ppo-lr64-training.npy','/tmp/ppo-minibatch-training.npy','/tmp/ppo-four-gpu-all-training.npy'],indices())])

def generate(kind,seed,count):
 start=time.perf_counter();cpu=time.process_time();idx=indices();forbidden=set(json.loads((R/'exclusions.json').read_text())['states']);a=Agent.load(source(seed));bot=CompletedSearch(a,ms=0,width=24,replies=8,depth=2);games=[];phis=[];cc=[];ys=[];hashes=[];stages=[];allvisited=set();i=0
 offset={'pilot':2420000000,'train':2440000000,'calibration':2450000000}[kind]+(seed-3101)*100000
 while len(games)<count:
  ps=offset+i;i+=1;s=prefix(ps);h=state_hash(s)
  if h in forbidden or any(contains(z,h) for z in idx):continue
  rng=random.Random(ps);moves=[];turns=[];localphi=[];localcodes=[];localhash=[];localphase=[]
  while not s.done:
   localphi.append(value_features(s).reshape(-1));localcodes.append(codes(s));localhash.append(state_hash(s));localphase.append(phase(s));turns.append(s.turn);m=bot.act(s,rng)
   moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
  assert all(x.get('target_complete',True) for x in bot.stats);bot.stats.clear()
  target=[outcome(s,p) for p in turns];phis+=localphi;cc+=localcodes;ys+=target;hashes+=localhash;stages+=localphase;allvisited.update(localhash);allvisited.add(state_hash(s))
  games.append(dict(prefix_seed=ps,seed=seed,seat=0,moves=moves,scores=[s.score(0),s.score(1)],turns=turns,targets=target,states=localhash))
  if len(games)%64==0:print(kind,seed,len(games),round(time.perf_counter()-start,1),flush=True)
 folder=R/f'{kind}-{seed}';folder.mkdir(exist_ok=True)
 with gzip.open(folder/'games.json.gz','wt') as f:json.dump(games,f)
 np.savez_compressed(folder/'data.npz',phi=np.asarray(phis),codes=np.asarray(cc,dtype=np.uint16),y=np.asarray(ys),hashes=np.array(hashes),phase=np.array(stages))
 np.save(folder/'visited.npy',np.array(sorted(bytes.fromhex(h) for h in allvisited),dtype='S32'))
 visits=np.bincount(np.asarray(cc).ravel(),minlength=SIZE)
 info=dict(kind=kind,seed=seed,games=len(games),states=len(ys),wall=time.perf_counter()-start,cpu=time.process_time()-cpu,rejected=i-count,source_sha256=sha(source(seed)),data_sha256=sha(folder/'data.npz'),visited_keys=int(np.count_nonzero(visits)),table_slots=SIZE,phase_states=np.bincount(stages,minlength=3).tolist(),terminal_wdl=np.unique(ys,return_counts=True)[1].tolist());(folder/'summary.json').write_text(json.dumps(info,indent=2)+'\n');print(info,flush=True)

def freeze():
 pilot=[json.loads((R/f'pilot-{s}/summary.json').read_text()) for s in SEEDS];count=1024 if max(x['wall']/x['games']*1024 for x in pilot)<=1200 else 512
 write('fixed-plan.json',dict(games_per_seed=count,calibration_games_per_seed=32,epochs=8,batch=256,lr=.01,workers=3,seeds=SEEDS,arms=['F','L','T'],pilot=pilot,criterion='1024 if max pilot extrapolated per-seed wall <=1200s else512',validation_prefixes=32,confirmation_prefixes=128));print('FIXED GAMES',count,flush=True)

def train(seed):
 cpu=time.process_time();start=time.perf_counter();d=dict(np.load(R/f'train-{seed}/data.npz'));source_agent=Agent.load(source(seed));summary=[]
 for arm in ('L','T'):
  t=Learner(source_agent,arm,seed+10000);logs=[]
  while True:
   row=t.update(d['phi'],d['codes'],d['y'])
   if row is None:break
   logs.append(row)
  t.save(R/f'{arm}-{seed}-state.json');t.agent().save(R/f'{arm}-{seed}.json',dict(seed=seed,source_sha256=sha(source(seed)),data_sha256=sha(R/f'train-{seed}/data.npz'),steps=t.step,epochs=t.epoch,objective='terminal WDL half MSE',policy_frozen=True));write(f'{arm}-{seed}-curve.json',logs);summary.append(dict(arm=arm,steps=t.step,epochs=t.epoch,finite=bool(np.isfinite(t.w).all()),visited_keys=int(np.count_nonzero(t.visits)),changed_parameters=int(np.count_nonzero(t.w-(source_agent.value_weights.reshape(-1) if arm=='L' else 0))),table_bytes=t.w.nbytes))
 write(f'train-summary-{seed}.json',dict(seed=seed,wall=time.perf_counter()-start,cpu=time.process_time()-cpu,arms=summary));print(seed,summary,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','generate','freeze','train']);p.add_argument('--kind',choices=['pilot','train','calibration']);p.add_argument('--seed',type=int,default=3101);p.add_argument('--games',type=int);a=p.parse_args()
 if a.mode=='prepare':prepare()
 elif a.mode=='freeze':freeze()
 elif a.mode=='train':train(a.seed)
 else:generate(a.kind,a.seed,a.games)
