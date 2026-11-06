"""Fresh paired prefix evaluation; identical search conditions for F/L/T."""
import argparse,gzip,hashlib,json,time,random,copy
from pathlib import Path
import numpy as np
from blokus.linear_value import Agent,value_features
from blokus.tuple_value import load_value,TupleAgent,codes,residual,phase,geometry,SIZE
from blokus.linear_tree import CompletedSearch
from blokus.linear_rl import prefix
from blokus.linear_train2 import state_hash
from scripts import linear_value_eval as base
from scripts import linear_search_run as search
from scripts import tuple_cycle as cycle
R=cycle.R;SEEDS=cycle.SEEDS;original_make=base.make

def source(label,seed):return cycle.source(seed) if label=='F' else R/f'{label}-{seed}.json'
def make(label,seed):
 if label=='gen3':return original_make(label,seed)
 return CompletedSearch(load_value(source(label,seed)),ms=100,width=24,replies=8,depth=2)
def model_sha(label,seed):return base.sha(f'artifacts/ppo-gen3/seed-{1751401+seed-3101}.pt') if label=='gen3' else base.sha(source(label,seed))
def setup():
 import torch
 torch.set_num_threads(1)
 assert all(base.sha(x['path'])==x['sha256'] for x in json.loads((R/'models.json').read_text()))
 base.R=R;base.make=make;base.model_sha=model_sha

def index():return np.load(R/'new-visited.npy',mmap_mode='r')
def prepare_eval():
 arrays=[np.load(p) for p in R.glob('*/visited.npy')];new=np.unique(np.concatenate(arrays));np.save(R/'new-visited.npy',new)
 cycle.write('training-summary.json',dict(new_states=len(new),raw_sha256=hashlib.sha256(new.tobytes()).hexdigest(),sources=[dict(path=str(p),sha256=base.sha(p)) for p in R.glob('*/visited.npy')]))
 cycle.write('trained-models.json',[dict(label=l,seed=s,path=str(source(l,s)),sha256=model_sha(l,s)) for l in ('F','L','T') for s in SEEDS])

def bank(phase_name,n):
 path=R/f'{phase_name}-prefixes.json'
 if path.exists():return json.loads(path.read_text())
 idx=cycle.indices()+[index()];exclude=set(json.loads((R/'exclusions.json').read_text())['states'])
 for p in R.glob('*-prefixes.json'):exclude.update(x['hash'] for x in json.loads(p.read_text())['prefixes'])
 rows=[];i=0;offset=2460000000 if phase_name=='validation' else 2470000000
 while len(rows)<n:
  ps=offset+i;i+=1;h=state_hash(prefix(ps))
  if h in exclude or any(base.contains(a,h) for a in idx):continue
  exclude.add(h);rows.append(dict(seed=ps,hash=h))
 d=dict(prefixes=rows,count=n,rejected=i-n,sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest());cycle.write(path.name,d);return d

def match(phase_name,left,right,seed,prefixes):
 path=R/f'{phase_name}-{left}-{right}-{seed}.json.gz'
 if path.exists():
  with gzip.open(path,'rt') as f:d=json.load(f)
  assert d['prefix_sha256']==prefixes['sha256'] and d['left_model_sha256']==model_sha(left,seed) and d['right_model_sha256']==model_sha(right,seed);return d
 d=base.match(phase_name,left,right,seed,prefixes);plain=R/f'{phase_name}-{left}-{right}-{seed}.json';raw=plain.read_bytes()
 with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:f.write(raw)
 assert gzip.decompress(path.read_bytes())==raw;plain.unlink();return d

def evaluate(phase_name):
 setup();prefixes=bank(phase_name,32 if phase_name=='validation' else 128)
 if phase_name=='validation':pairs=[('T','L'),('T','F'),('L','F'),('F','gen3'),('L','gen3'),('T','gen3')]
 else:
  sel=json.loads((R/'selection.json').read_text());assert sel['proceed'];assert all(base.sha(x['path'])==x['sha256'] for x in sel['hashes']);pairs=[('T','L'),('T','F'),('T','gen3')]
 summary={};costs={}
 for l,r in pairs:
  ds=[match(phase_name,l,r,s,prefixes) for s in SEEDS];summary[l+'/'+r]=base.cluster(ds);summary[l+'/'+r]['wall_seconds']=sum(d['wall'] for d in ds);costs[l+'/'+r]={side:search.summarize_stats(sum([d[side+'_search'] for d in ds],[])) for side in ('left','right') if ds[0][side+'_search']};cycle.write(phase_name+'-summary.json',summary);cycle.write(phase_name+'-costs.json',costs)
 if phase_name=='validation':
  st=costs['T/gen3']['left'];gate=summary['T/L']['point']>=.5 and summary['T/F']['point']>=.5 and summary['T/gen3']['point']>=summary['F/gen3']['point']-.02 and st['target_complete']>=.8 and st['greedy_fallback']<.1
  cycle.write('selection.json',dict(proceed=gate,selected='T' if gate else None,criteria=dict(TL=summary['T/L']['point'],TF=summary['T/F']['point'],Tgen3=summary['T/gen3']['point'],Fgen3=summary['F/gen3']['point'],complete=st['target_complete'],fallback=st['greedy_fallback']),hashes=json.loads((R/'trained-models.json').read_text()),seed_selection=False,endpoint_selection=False));print('GATE',gate,flush=True)

def calibration():
 rows=[]
 canon=geometry()[3];reachable=np.concatenate([np.unique(canon)+729*i for i in range(6)])
 for s in SEEDS:
  train=dict(np.load(R/f'train-{s}/data.npz'));dev=dict(np.load(R/f'calibration-{s}/data.npz'));counts=np.bincount(train['codes'].ravel(),minlength=SIZE);missing=counts[dev['codes']]==0
  tables=dict(seed=s,allocated_keys=SIZE,canonical_keys=len(reachable),visited_keys=int(np.count_nonzero(counts)),canonical_visit_fraction=float(np.mean(counts[reachable]>0)),unseen_dev_occurrence_fraction=float(missing.mean()),distinct_training_raw_canonical_keys=len(np.unique(train['codes'])),intentional_sharing='translations, transpose, reverse-code180, player antisymmetry; no hashing collisions',bytes=SIZE*8)
  stats={}
  for label in ('F','L','T'):
   a=load_value(source(label,s));datasets={}
   for name,d in [('train',train),('unused_games',dev)]:
    raw=d['phi']@a.value_weights.reshape(-1)
    if isinstance(a,TupleAgent):raw+=residual(a.table,d['codes'])
    pred=np.clip(raw,-1,1);byphase={}
    for p in range(3):
     ix=d['phase']==p;target=d['y'][ix];v=pred[ix];bins=[]
     for lo,hi in [(-1,-.5),(-.5,0),(0,.5),(.5,1.00001)]:
      z=(v>=lo)&(v<hi);bins.append(dict(range=[lo,hi],count=int(z.sum()),prediction=float(v[z].mean()) if z.any() else None,target=float(target[z].mean()) if z.any() else None))
     byphase[str(p)]=dict(states=int(ix.sum()),mse=float(np.mean((v-target)**2)),sign_accuracy=float(np.mean(np.sign(v)==target)),prediction_quantiles=np.quantile(v,[0,.1,.5,.9,1]).tolist(),bins=bins)
    datasets[name]=dict(saturation_fraction=float(np.mean(np.abs(pred)>=1)),mse=float(np.mean((pred-d['y'])**2)),raw_mse=float(np.mean((raw-d['y'])**2)),by_phase=byphase)
   stats[label]=datasets
  rows.append(dict(seed=s,coverage=tables,models=stats))
 cycle.write('calibration.json',rows);print('CALIBRATION SAVED',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','calibration','validation','confirmation']);a=p.parse_args()
 if a.mode=='prepare':prepare_eval()
 elif a.mode=='calibration':calibration()
 else:evaluate(a.mode)
