"""Frozen hybrid width/depth experiment; no training or checkpoint selection."""
import argparse,copy,gzip,hashlib,json,random,time
from pathlib import Path
import numpy as np
from blokus.engine import State,Move,legal_moves,play
from blokus.linear_rl import prefix
from blokus.linear_train2 import state_hash
from blokus.linear_value import Agent,Search
from blokus.linear_tree import CompletedSearch
from scripts import linear_value_eval as base
from scripts.linear_hybrid_run import cluster
R=Path('artifacts/linear-search');SEEDS=(3101,3102,3103)
def write(n,d):(R/n).write_text(json.dumps(d,indent=2)+'\n')
def source(s):return Path(f'artifacts/linear-hybrid/exports/seed-{s}.json')
def index():
 a=np.load('/tmp/linear-retrain-training.npy',mmap_mode='r');assert hashlib.sha256(a.tobytes()).hexdigest()=='004c40ed84b7de0ab07372006298735b435f99f16a4065d503ed231592850798';return a

def prepare():
 R.mkdir(exist_ok=True);rows=[]
 for s in SEEDS:
  a=Agent.load(source(s));p=Agent.load(f'artifacts/linear-value/old-{s}/policy.json');v=Agent.load(f'artifacts/linear-value/new-{s}/policy.json');assert np.array_equal(a.weights,p.weights) and np.array_equal(a.value_weights,v.value_weights)
  for path in (source(s),Path(f'artifacts/linear-value/old-{s}/policy.json'),Path(f'artifacts/linear-value/new-{s}/policy.json'),Path(f'artifacts/ppo-gen3/seed-{1751401+s-3101}.pt')):rows.append(dict(path=str(path),sha256=base.sha(path)))
 write('models.json',rows)
 forbidden=set();provenance=[]
 for folder in (Path('artifacts/linear-value'),Path('artifacts/linear-hybrid'),Path('/home/eletim/blokus-linear-retrain/artifacts/linear-value-retrain')):
  for p in list(folder.glob('*prefixes.json'))+list(folder.glob('exclusions.json')):
   d=json.loads(p.read_text());forbidden.update(d.get('states',[]));forbidden.update(x['hash'] for x in d.get('prefixes',[]));provenance.append(dict(path=str(p),sha256=base.sha(p)))
  for p in list(folder.glob('*calibration.json'))+list(folder.glob('diagnostic.json.gz')):
   with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:d=json.load(f)
   forbidden.update(state_hash(prefix(g['prefix_seed'])) for g in d.get('games',[]) if 'prefix_seed' in g);provenance.append(dict(path=str(p),sha256=base.sha(p)))
 write('exclusions.json',dict(states=sorted(forbidden),sources=provenance,training_states=len(index()),training_sha256=hashlib.sha256(index().tobytes()).hexdigest()))

def bank(phase,n):
 p=R/f'{phase}-prefixes.json'
 if p.exists():return json.loads(p.read_text())
 a=index();excluded=set(json.loads((R/'exclusions.json').read_text())['states'])
 for q in R.glob('*-prefixes.json'):excluded.update(x['hash'] for x in json.loads(q.read_text())['prefixes'])
 rows=[];i=0;offset={'dev':2380000000,'validation':2390000000,'confirmation':2400000000}[phase]
 while len(rows)<n:
  seed=offset+i;i+=1;h=state_hash(prefix(seed))
  if h in excluded or base.contains(a,h):continue
  excluded.add(h);rows.append(dict(seed=seed,hash=h))
 d=dict(prefixes=rows,count=n,rejected=i-n,sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest());write(p.name,d);return d

def make(label,seed):
 if label=='gen3':return original_make(label,seed)
 agent=Agent.load(source(seed))
 if label=='old6':return Search(agent,ms=6)
 ms=json.loads((R/'settings.json').read_text())['ms']
 return CompletedSearch(agent,ms=ms,width=24 if label=='B' else 6,replies=8 if label=='B' else 3,depth=4 if label=='C' else 2)
original_make=base.make

def model_sha(label,seed):
 path=f'artifacts/ppo-gen3/seed-{1751401+seed-3101}.pt' if label=='gen3' else source(seed)
 return hashlib.sha256((base.sha(path)+label+(R/'settings.json').read_text()).encode()).hexdigest()
def setup():
 import torch
 torch.set_num_threads(1)
 assert all(base.sha(x['path'])==x['sha256'] for x in json.loads((R/'models.json').read_text()))
 base.R=R;base.make=make;base.model_sha=model_sha

def pilot():
 start=time.perf_counter();aidx=index();games=[];states=[]
 for i,row in enumerate(bank('dev',24)['prefixes']):
  s=prefix(row['seed']);a=Agent.load(source(SEEDS[i%3]));rng=random.Random(row['seed']);moves=[];ply=4
  while not s.done:
   if ply in (8,18,28) and not base.contains(aidx,state_hash(s)):states.append(dict(seed=SEEDS[i%3],game=row['seed'],ply=ply,state=copy.deepcopy([s.boards,s.remaining,s.last,s.turn,s.passes]),hash=state_hash(s)))
   m=a.act(s,rng);moves.append(None if m is None else list((m.piece,m.orientation,m.row,m.col)));play(s,m,validate=False);ply+=1
  games.append(dict(prefix_seed=row['seed'],seat=0,moves=moves,scores=[s.score(0),s.score(1)]))
 rows=[];timing=[];micro=[]
 for row in states:
  s=State(*copy.deepcopy(row['state']));assert state_hash(s)==row['hash'];a=Agent.load(source(row['seed']));rr={}
  for w,r in ((6,3),(24,3),(6,8),(24,8)):
   b=CompletedSearch(a,ms=0,width=w,replies=r,base_width=w,base_replies=r);m=b.act(s,random.Random(0));rr[f'{w}x{r}']=dict(move=list((m.piece,m.orientation,m.row,m.col)) if m else None,stats=b.stats[-1])
  rows.append(dict(**row,results=rr))
  for ms in (25,100):
   for label,w,r,d in (('A',6,3,2),('B',24,8,2),('C',6,3,4)):
    b=CompletedSearch(a,ms=ms,width=w,replies=r,depth=d)
    for warm in (False,True):b.act(s,random.Random(0));timing.append(dict(label=label,budget=ms,warm=warm,game=row['game'],ply=row['ply'],stats=b.stats[-1]))
  # Standalone costs are overlapping alternatives, not additive fused scan breakdown.
  from blokus.ai import features
  q=time.perf_counter();qs=legal_moves(s);legal=(time.perf_counter()-q)*1000
  q=time.perf_counter();features(s,qs);feat=(time.perf_counter()-q)*1000
  q=time.perf_counter();a.inputs(s);fused=(time.perf_counter()-q)*1000
  q=time.perf_counter();a.value(s);value=(time.perf_counter()-q)*1000
  micro.append(dict(ply=row['ply'],legal_python_ms=legal,features_python_ms=feat,native_fused_ms=fused,value_ms=value,candidates=len(qs)))
 rates={str(ms):{label:float(np.mean([x['stats'].get('target_complete',True) for x in timing if x['budget']==ms and x['label']==label and x['warm']])) for label in ('A','B','C')} for ms in (25,100)}
 budget=25 if min(rates['25']['B'],rates['25']['C'])>=.9 else 100
 write('settings.json',dict(ms=budget,A=[6,3,2],B=[24,8,2],C=[6,3,4],rule='first 25ms if B,C target completion >=.90 else100',pilot_rates=rates))
 with gzip.open(R/'pilot.json.gz','wt') as f:json.dump(dict(games=games,states=rows,timing=timing,micro=micro,wall=time.perf_counter()-start),f)
 changes={key:float(np.mean([x['results'][key]['move']!=x['results']['6x3']['move'] for x in rows])) for key in ('24x3','6x8','24x8')}
 outside=float(np.mean([x['results']['24x8']['move'] not in x['results']['6x3']['stats'].get('roots',[]) for x in rows if x['results']['24x8']['move'] is not None]))
 write('pilot-summary.json',dict(games=len(games),states=len(rows),choice_change=changes,wide_choice_outside_narrow=outside,completion=rates,budget=budget,wall=time.perf_counter()-start));print('PILOT',budget,changes,rates,flush=True)

def read_match(path):
 with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:return json.load(f)

def match(phase,left,right,seed,prefixes):
 path=R/f'{phase}-{left}-{right}-{seed}.json.gz'
 if path.exists():
  d=read_match(path)
  assert d['left_model_sha256']==model_sha(left,seed) and d['right_model_sha256']==model_sha(right,seed) and d['prefix_sha256']==prefixes['sha256']
  return d
 return base.match(phase,left,right,seed,prefixes)

def archive():
 rows=[]
 for phase in ('validation','confirmation'):
  for path in R.glob(f'{phase}-*.json'):
   d=json.loads(path.read_text())
   if not isinstance(d.get('games'),list):continue
   data=path.read_bytes();dest=Path(str(path)+'.gz')
   with gzip.GzipFile(filename=str(dest),mode='wb',mtime=0) as f:f.write(data)
   assert gzip.decompress(dest.read_bytes())==data
   path.unlink()
  for dest in sorted(R.glob(f'{phase}-*.json.gz')):
   data=gzip.decompress(dest.read_bytes());d=json.loads(data)
   if not isinstance(d.get('games'),list):continue
   rows.append(dict(path=str(dest),raw_sha256=hashlib.sha256(data).hexdigest(),compressed_sha256=base.sha(dest),games=len(d['games'])))
 write('archive-manifest.json',rows)

def summarize_stats(stats):
 s=[x for x in stats if not x.get('forced_pass')]
 return dict(moves=len(s),target_complete=float(np.mean([x.get('target_complete',x['complete']) for x in s])),greedy_fallback=float(np.mean([x['fallback'] for x in s])),ms_quantiles=np.quantile([x['ms'] for x in s],[.5,.9,.99,1]).tolist(),overshoots=sum(bool(x.get('overshoot')) for x in s),depths={str(d):sum(x.get('depth',2 if x['complete'] else 0)==d for x in s) for d in (0,2,4)},mean_nodes=float(np.mean([x['nodes'] for x in s])),stage_fallback=float(np.mean([x['complete'] and not x.get('target_complete',x['complete']) for x in s])),overshoot_fraction=float(np.mean([bool(x.get('overshoot')) for x in s])),overshoot_ms_quantiles=np.quantile([x.get('overshoot_ms',0) for x in s],[.5,.9,.99,1]).tolist(),mean_completed_roots=float(np.mean([x.get('completed_roots',0) for x in s])),mean_reply_edges=float(np.mean([x.get('replies_evaluated',0) for x in s])),mean_input_ms=float(np.mean([x.get('input_ms',0) for x in s])),mean_value_ms=float(np.mean([x.get('value_ms',0) for x in s])))

def evaluate(phase):
 setup();prefixes=bank(phase,32 if phase=='validation' else 128)
 if phase=='validation':pairs=[('A','gen3'),('B','gen3'),('C','gen3'),('B','A'),('C','A'),('old6','gen3')]
 else:
  sel=json.loads((R/'selection.json').read_text());assert sel['selected'];pairs=[(sel['selected'],'A'),(sel['selected'],'gen3')]
 result={};stats={}
 for left,right in pairs:
  matches=[match(phase,left,right,s,prefixes) for s in SEEDS];key=f'{left}/{right}';result[key]=cluster(matches)
  stats[key]={side:summarize_stats(sum([d[side+'_search'] for d in matches],[])) for side in ('left','right') if matches[0][side+'_search']}
  write(f'{phase}-summary.json',result);write(f'{phase}-costs.json',stats)
 if phase=='validation':
  choices=[];gates={}
  for c in ('B','C'):
   t=stats[c+'/gen3']['left'];g=result[c+'/gen3']['point'];direct=result[c+'/A']['point'];ok=direct>=.5 and g>=result['A/gen3']['point']-.02 and t['target_complete']>=.8 and t['greedy_fallback']<.1
   gates[c]=dict(pass_=ok,direct=direct,gen3=g,completion=t['target_complete'],fallback=t['greedy_fallback'])
   if ok:choices.append((g,direct,-result[c+'/gen3']['left_ms'],c=='B',c))
  selected=max(choices)[-1] if choices else None
  write('selection.json',dict(selected=selected,gates=gates,settings_sha256=base.sha(R/'settings.json'),model_hashes=json.loads((R/'models.json').read_text()),checkpoint_seed_selection=False));print('SELECTED',selected,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','pilot','validation','confirmation','archive']);args=p.parse_args()
 if args.mode=='prepare':prepare()
 elif args.mode=='pilot':pilot()
 elif args.mode=='archive':archive()
 else:evaluate(args.mode)
