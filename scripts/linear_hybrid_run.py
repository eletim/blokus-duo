"""Fixed-model mixed policy/value experiment; original results remain untouched."""
import argparse,copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
from scripts import linear_value_eval as base
from blokus.linear_value import Agent,Search,outcome
from blokus.linear_train2 import state_hash
from blokus.linear_rl import prefix
from blokus.engine import Move,play,is_legal,legal_moves
R=Path('artifacts/linear-hybrid');OLD=Path('artifacts/linear-value');SEEDS=(3101,3102,3103)
def write(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
def locked():
 rows=json.loads((R/'models.json').read_text())
 assert all(base.sha(r['path'])==r['sha256'] for r in rows)
 return rows

def make(label,seed):
 if label=='gen3':
  from blokus.ppo_policy import PPOAgent
  return PPOAgent(model_seed=1751401+seed-3101,device='cpu',backend='native')
 if label=='old':return Agent.load(OLD/f'old-{seed}/policy.json')
 if label=='previous':return Search(Agent.load(OLD/f'new-{seed}/policy.json'),ms=6)
 if label=='hybrid':return Search(Agent.load_pair(OLD/f'old-{seed}/policy.json',OLD/f'new-{seed}/policy.json'),ms=6)
 if label=='retrained':return Search(Agent.load(R/f'retrained-{seed}.json'),ms=6)
 raise ValueError(label)
def model_sha(label,seed):
 if label=='gen3':return base.sha(f'artifacts/ppo-gen3/seed-{1751401+seed-3101}.pt')
 if label=='old':return base.sha(OLD/f'old-{seed}/policy.json')
 if label=='previous':return base.sha(OLD/f'new-{seed}/policy.json')
 if label=='retrained':return base.sha(R/f'retrained-{seed}.json')
 return hashlib.sha256((model_sha('old',seed)+model_sha('previous',seed)).encode()).hexdigest()

def index():
 a=np.load('/tmp/linear-value-training.npy',mmap_mode='r')
 assert hashlib.sha256(a.tobytes()).hexdigest()==json.loads((OLD/'training-summary.json').read_text())['state_sha256']
 return a

def bank(phase,count):
 p=R/f'{phase}-prefixes.json'
 if p.exists():return json.loads(p.read_text())
 a=index();forbidden=set(json.loads((OLD/'exclusions.json').read_text())['states'])
 for phase0 in ('validation','confirmation'):
  forbidden.update(x['hash'] for x in json.loads((OLD/f'{phase0}-prefixes.json').read_text())['prefixes'])
 forbidden.update(state_hash(prefix(g['prefix_seed'])) for g in json.loads((OLD/'value-calibration.json').read_text())['games'])
 for p0 in R.glob('*-prefixes.json'):forbidden.update(x['hash'] for x in json.loads(p0.read_text())['prefixes'])
 if (R/'calibration.json').exists():forbidden.update(state_hash(prefix(g['prefix_seed'])) for g in json.loads((R/'calibration.json').read_text())['games'])
 rows=[];i=0;start=2240000000 if phase=='validation' else 2250000000
 while len(rows)<count:
  ps=start+i;i+=1;h=state_hash(prefix(ps))
  if h in forbidden or base.contains(a,h):continue
  forbidden.add(h);rows.append(dict(seed=ps,hash=h))
 d=dict(prefixes=rows,count=count,rejected=i-count,sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),training_sha256=hashlib.sha256(a.tobytes()).hexdigest());write(p.name,d);return d

def cluster(matches):
 d=base.cluster(matches);groups={}
 for m in matches:
  for g in m['games']:groups.setdefault(g['prefix_seed'],[]).append(g['point'])
 points=np.array([np.mean(x) for x in groups.values()]);ix=np.random.default_rng(22800).integers(len(groups),size=(10000,len(groups)))
 d.pop('point_ci_bonferroni3');d['point_ci_bonferroni2']=np.quantile(points[ix].mean(1),[.0125,.9875]).tolist();d['wall_seconds']=sum(m['wall'] for m in matches)
 return d

def setup():
 import torch
 torch.set_num_threads(1);locked();base.R=R;base.make=make;base.model_sha=model_sha

def evaluate(phase):
 setup();prefixes=bank(phase,64 if phase=='validation' else 128)
 if phase=='validation':pairs=[('hybrid','previous'),('hybrid','old'),('hybrid','gen3'),('previous','gen3')]
 else:
  lock=json.loads((R/'selection.json').read_text());assert lock['proceed'];assert all(model_sha(label,s)==h for label,seedhash in lock['hashes'].items() for s0,h in seedhash.items() for s in [int(s0)])
  selected=lock['selected'];pairs=[(selected,'previous'),(selected,'gen3'),(selected,'old')]
 summary={}
 for left,right in pairs:
  summary[f'{left}/{right}']=cluster([base.match(phase,left,right,s,prefixes) for s in SEEDS]);write(f'{phase}-summary.json',summary)
 if phase=='validation':
  stats=[]
  for s in SEEDS:stats+=json.loads((R/f'validation-hybrid-previous-{s}.json').read_text())['left_search']
  fallback=float(np.mean([x['fallback'] for x in stats]));gate=fallback<.1 and summary['hybrid/previous']['point']>=.48 and summary['hybrid/gen3']['point']>=summary['previous/gen3']['point']-.03
  write('validation-gate.json',dict(gate=gate,fallback=fallback,needs_distribution_diagnostic=summary['hybrid/previous']['point']<.5 or not gate));print('GATE',gate,flush=True)

def select():
 locked();g=json.loads((R/'validation-gate.json').read_text());assert not g['needs_distribution_diagnostic'],'run prescribed distribution diagnostic before selection'
 hashes={label:{str(s):model_sha(label,s) for s in SEEDS} for label in ('hybrid','previous','old','gen3')}
 write('selection.json',dict(proceed=g['gate'],selected='hybrid',checkpoint_selection=False,seed_selection=False,hashes=hashes,source='predeclared fixed hybrid; no further training'))

def audit():
 setup();a=index();rows=[]
 for phase in ('validation','confirmation'):
  data=json.loads((R/f'{phase}-prefixes.json').read_text());starts={r['seed']:r['hash'] for r in data['prefixes']};assert len(starts)==data['count']
  assert data['sha256']==hashlib.sha256(json.dumps(data['prefixes'],sort_keys=True).encode()).hexdigest()
  expected=json.loads((R/f'{phase}-summary.json').read_text());allmatches={}
  for p in R.glob(f'{phase}-*.json'):
   d=json.loads(p.read_text())
   if not isinstance(d.get('games'),list):continue
   assert d['prefix_sha256']==data['sha256'];assert d['left_model_sha256']==model_sha(d['left'],d['seed']) and d['right_model_sha256']==model_sha(d['right'],d['seed'])
   overlap=0;seats=set()
   for stat in d['left_search']+d['right_search']:
    if stat.get('forced_pass'):continue
    if stat['complete']:assert stat['completed_roots']==stat['root_candidates'] and len(stat['values'])==stat['root_candidates']
    if stat['fallback']:assert not stat['values']
   for g in d['games']:
    s=prefix(g['prefix_seed']);assert state_hash(s)==starts[g['prefix_seed']];assert (g['prefix_seed'],g['seat']) not in seats;seats.add((g['prefix_seed'],g['seat']))
    for row in g['moves']:
     overlap+=base.contains(a,state_hash(s));m=Move(*row) if row else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False)
    overlap+=base.contains(a,state_hash(s));assert s.done and [s.score(0),s.score(1)]==g['scores'];diff=s.score(g['seat'])-s.score(1-g['seat']);assert g['difference']==diff and g['point']==np.sign(diff)*.5+.5
   assert overlap==0 and len(seats)==data['count']*2
   allmatches.setdefault(f"{d['left']}/{d['right']}",[]).append(d);rows.append(dict(path=str(p),games=len(d['games']),overlap=int(overlap)))
  for k,matches in allmatches.items():
   matches.sort(key=lambda x:x['seed']);assert cluster(matches)==expected[k]
 val={x['hash'] for x in json.loads((R/'validation-prefixes.json').read_text())['prefixes']};conf={x['hash'] for x in json.loads((R/'confirmation-prefixes.json').read_text())['prefixes']};assert not val&conf
 write('audit.json',dict(pass_=True,rows=rows,all_games_replayed=True,arithmetic=True,model_hashes=True,training_overlap=0,selection_confirmation_disjoint=True));print('AUDIT PASS',sum(x['games'] for x in rows),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','select','confirmation','audit']);args=p.parse_args()
 if args.mode=='select':select()
 elif args.mode=='audit':audit()
 else:evaluate(args.mode)
