"""Fixed endpoint evaluation; paired prefixes, all seeds clustered together."""
import argparse,copy,json,random,time,hashlib
from pathlib import Path
import numpy as np
from blokus.linear_rl import prefix
from blokus.linear_train2 import state_hash
from blokus.linear_value import Agent,Search,outcome
from blokus.engine import State,Move,play,is_legal,legal_moves
R=Path('artifacts/linear-value');SEEDS=[3101,3102,3103]
def write(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def contains(index,h):
 v=np.bytes_(bytes.fromhex(h));i=np.searchsorted(index,v);return i<len(index) and index[i]==v

def calibration():
 rows=[];games=[];index=np.load('/tmp/linear-value-training.npy',mmap_mode='r')
 for seed in SEEDS:
  a=Agent.load(R/f'new-{seed}/policy.json');initial=Agent();rng=np.random.default_rng(2210000+seed);pred=[];labels=[]
  for i in range(64):
   ps=2210000000+seed*10000+i;s=prefix(ps);assert not contains(index,state_hash(s));seat=i%2;kind=(i//2)%4;values=[];moves=[]
   while not s.done:
    if s.turn==seat:values.append(a.value(s))
    bot=initial if kind==3 and s.turn!=seat else a;m=bot.act(s,rng) if kind==3 and s.turn!=seat else bot.sample(s,rng)[0]
    moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   y=outcome(s,seat);pred.extend(values);labels.extend([y]*len(values));games.append(dict(seed=seed,prefix_seed=ps,seat=seat,moves=moves,scores=[s.score(0),s.score(1)],predictions=values,reward=y))
  rows.append(dict(seed=seed,states=len(pred),mse=float(np.mean((np.array(pred)-labels)**2)),zero_mse=float(np.mean(np.square(labels)))))
 gate=np.mean([x['mse'] for x in rows])<np.mean([x['zero_mse'] for x in rows]);write('value-calibration.json',dict(rows=rows,gate=bool(gate),games=games,note='new games under sampled policy/initial pool; past approximated by final current, not independent minimax truth'));print('VALUE GATE',gate,rows,flush=True)

def bank(phase,count):
 path=R/f'{phase}-prefixes.json'
 if path.exists():return json.loads(path.read_text())
 index=np.load('/tmp/linear-value-training.npy',mmap_mode='r');forbidden=set(json.loads((R/'exclusions.json').read_text())['states'])
 for g in json.loads((R/'value-calibration.json').read_text())['games']:forbidden.add(state_hash(prefix(g['prefix_seed'])))
 if phase=='confirmation':forbidden.update(p['hash'] for p in json.loads((R/'validation-prefixes.json').read_text())['prefixes'])
 base=2220000000 if phase=='validation' else 2230000000;rows=[];i=0;seen=set()
 while len(rows)<count:
  ps=base+i;i+=1;s=prefix(ps);h=state_hash(s)
  if h in forbidden or h in seen or contains(index,h):continue
  seen.add(h);rows.append(dict(seed=ps,hash=h))
 d=dict(prefixes=rows,count=count,rejected=i-count,sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),training_sha256=json.loads((R/'training-summary.json').read_text())['state_sha256']);write(path.name,d);return d

def make(label,seed):
 if label=='gen3':
  from blokus.ppo_policy import PPOAgent
  return PPOAgent(model_seed=1751401+seed-3101,device='cpu',backend='native')
 if label=='initial':return Agent('old')
 if label in ('old','new'):return Agent.load(R/f'{label}-{seed}/policy.json')
 if label.startswith('search'):
  parts=label.split('-');ms=int(parts[1]);return Search(Agent.load(R/f'new-{seed}/policy.json'),ms=ms,learned=len(parts)==2)
 raise ValueError(label)

def model_sha(label,seed):
 if label=='initial':return hashlib.sha256(Agent('old').weights.tobytes()).hexdigest()
 if label=='gen3':return sha(f'artifacts/ppo-gen3/seed-{1751401+seed-3101}.pt')
 mode='new' if label.startswith('search') else label
 return sha(R/f'{mode}-{seed}/policy.json')

def match(phase,left,right,seed,prefixes):
 path=R/f'{phase}-{left}-{right}-{seed}.json'
 if path.exists():
  d=json.loads(path.read_text());assert d['left_model_sha256']==model_sha(left,seed) and d['right_model_sha256']==model_sha(right,seed) and d['prefix_sha256']==prefixes['sha256'];return d
 a=make(left,seed);b=make(right,seed)
 for bot in (a,b):
  bot.act(prefix(prefixes['prefixes'][0]['seed']),random.Random(0))
  if isinstance(bot,Search):bot.stats.clear()
 games=[];times=[[],[]];start=time.perf_counter()
 for row in prefixes['prefixes']:
  for seat in (0,1):
   s=prefix(row['seed']);rng=random.Random(row['seed']*2+seat);moves=[]
   while not s.done:
    side=0 if s.turn==seat else 1;begin=time.perf_counter();m=(a if side==0 else b).act(s,rng);times[side].append((time.perf_counter()-begin)*1000)
    moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   diff=s.score(seat)-s.score(1-seat);games.append(dict(prefix_seed=row['seed'],seat=seat,moves=moves,scores=[s.score(0),s.score(1)],difference=diff,point=float(np.sign(diff)*.5+.5)))
 d=dict(left_model_sha256=model_sha(left,seed),right_model_sha256=model_sha(right,seed),warmup_excluded=True,left=left,right=right,seed=seed,prefix_sha256=prefixes['sha256'],games=games,wall=time.perf_counter()-start,left_ms=float(np.mean(times[0])),right_ms=float(np.mean(times[1])),left_search=a.stats if isinstance(a,Search) else [],right_search=b.stats if isinstance(b,Search) else []);write(path.name,d);print(phase,left,right,seed,round(np.mean([g['point'] for g in games]),3),flush=True);return d

def cluster(matches):
 groups={}
 for d in matches:
  for g in d['games']:groups.setdefault(g['prefix_seed'],[]).append(g)
 assert all(len(g)==6 for g in groups.values());points=np.array([np.mean([x['point'] for x in g]) for g in groups.values()]);diff=np.array([np.mean([x['difference'] for x in g]) for g in groups.values()]);ix=np.random.default_rng(22800).integers(len(groups),size=(10000,len(groups)))
 return dict(point=float(points.mean()),point_ci95=np.quantile(points[ix].mean(1),[.025,.975]).tolist(),point_ci_bonferroni3=np.quantile(points[ix].mean(1),[.05/6,1-.05/6]).tolist(),difference=float(diff.mean()),difference_ci95=np.quantile(diff[ix].mean(1),[.025,.975]).tolist(),games=sum(len(d['games']) for d in matches),prefixes=len(groups),by_seed={str(d['seed']):dict(point=float(np.mean([g['point'] for g in d['games']])),difference=float(np.mean([g['difference'] for g in d['games']]))) for d in matches},left_ms=float(np.mean([d['left_ms'] for d in matches])),right_ms=float(np.mean([d['right_ms'] for d in matches])))

def evaluate(phase):
 import torch
 torch.set_num_threads(1)
 prefixes=bank(phase,64 if phase=='validation' else 128);summary={}
 if phase=='validation':
  pairs=[('old','initial'),('new','initial'),('new','old'),('old','gen3'),('new','gen3')]
  if json.loads((R/'value-calibration.json').read_text())['gate']:pairs += [('search-6','new'),('search-25','new')]
 else:
  lock=json.loads((R/'selection.json').read_text());assert all(sha(p)==h for p,h in lock['model_hashes'].items());pairs=[('new','old'),('new','initial'),('new','gen3')]
  if lock['search']:
   label=lock['search'];pairs += [(label,'new'),(label,label+'-static'),(label,'gen3')]
 allmatches={}
 for a,b in pairs:
  ms=[match(phase,a,b,s,prefixes) for s in SEEDS];key=f'{a}/{b}';summary[key]=cluster(ms);allmatches[key]=ms
 write(f'{phase}-summary.json',summary)
 if phase=='validation':
  choices=[]
  for budget in (6,25):
   key=f'search-{budget}/new'
   if key in summary:
    stats=[r for d in allmatches[key] for r in d['left_search']];fallback=np.mean([x['fallback'] for x in stats]);point=summary[key]['point']
    if point>=.5 and fallback<.5:choices.append((point,-budget,f'search-{budget}'))
  selected=max(choices)[2] if choices else None;hashes={str(R/f'{mode}-{s}/policy.json'):sha(R/f'{mode}-{s}/policy.json') for mode in ('old','new') for s in SEEDS};hashes.update({f'artifacts/ppo-gen3/seed-{1751401+i}.pt':sha(f'artifacts/ppo-gen3/seed-{1751401+i}.pt') for i in range(3)})
  write('selection.json',dict(search=selected,model_hashes=hashes,checkpoint='fixed endpoint',seed_selection=False));print('SEARCH SELECTED',selected,flush=True)

def audit():
 index=np.load('/tmp/linear-value-training.npy',mmap_mode='r');rows=[]
 for phase in ('validation','confirmation'):
  bankdata=json.loads((R/f'{phase}-prefixes.json').read_text());starts={r['seed']:r['hash'] for r in bankdata['prefixes']};assert len(starts)==bankdata['count']
  for p in R.glob(f'{phase}-*.json'):
   d=json.loads(p.read_text())
   if 'games' not in d:continue
   assert d['prefix_sha256']==bankdata['sha256'];assert d['left_model_sha256']==model_sha(d['left'],d['seed']) and d['right_model_sha256']==model_sha(d['right'],d['seed']);overlap=0;pairs=set()
   for stat in d['left_search']+d['right_search']:
    if not stat.get('forced_pass') and stat['complete']:assert stat['completed_roots']==stat['root_candidates'] and len(stat['values'])==stat['root_candidates']
    if stat['fallback']:assert not stat['values']
   for g in d['games']:
    s=prefix(g['prefix_seed']);assert state_hash(s)==starts[g['prefix_seed']] and not contains(index,state_hash(s));pairs.add((g['prefix_seed'],g['seat']))
    for move in g['moves']:
     overlap+=contains(index,state_hash(s));m=Move(*move) if move else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False)
    overlap+=contains(index,state_hash(s));assert s.done and g['scores']==[s.score(0),s.score(1)];diff=s.score(g['seat'])-s.score(1-g['seat']);assert diff==g['difference'] and g['point']==np.sign(diff)*.5+.5
   assert len(pairs)==bankdata['count']*2;rows.append(dict(path=str(p),games=len(d['games']),train_overlap_visits=int(overlap)))
 # Independently replay calibration terminal labels and state-view predictions.
 calibration=json.loads((R/'value-calibration.json').read_text());models={seed:Agent.load(R/f'new-{seed}/policy.json') for seed in SEEDS}
 for game in calibration['games']:
  s=prefix(game['prefix_seed']);pred=[]
  for row in game['moves']:
   if s.turn==game['seat']:pred.append(models[game['seed']].value(s))
   move=Move(*row) if row else None;assert is_legal(s,move) if move else not legal_moves(s);play(s,move,validate=False)
  assert s.done and game['scores']==[s.score(0),s.score(1)] and game['reward']==outcome(s,game['seat']) and np.allclose(pred,game['predictions'],atol=1e-12,rtol=0)
 write('evaluation-audit.json',dict(rows=rows,pass_=True,all_games_replayed=True,calibration_labels_replayed=len(calibration['games'])));print('EVAL AUDIT',sum(x['train_overlap_visits'] for x in rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibration','validation','confirmation','audit']);a=p.parse_args()
 if a.mode=='calibration':calibration()
 elif a.mode=='audit':audit()
 else:evaluate(a.mode)
