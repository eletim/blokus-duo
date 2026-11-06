"""Frozen checkpoint evaluation; shared disjoint prefixes and prefix/three-seed bootstrap."""
import argparse,copy,json,random,time,hashlib
from pathlib import Path
import numpy as np
from .ppo_policy import PPOAgent
from .ppo_native import observe
from .ai import heuristic,seed_all
from .engine import State,Move,play,is_legal,legal_moves
from .ppo_runtime import state_hash
from .ppo_gpu_audit import R,CACHE,write
from .ppo_gpu_train import sha
from .tournament import prefix_state,digest,read_prefixes,summarize
SEEDS=[1751401,1751402,1751403]
def contains(index,h):
 v=np.bytes_(bytes.fromhex(h));i=np.searchsorted(index,v);return i<len(index) and index[i]==v

def bank(phase,count,seed):
 path=R/f'{phase}-prefixes.json'
 if path.exists():
  data=read_prefixes(path);assert data['training_union_sha256']==json.loads((R/'training-audit.json').read_text())['union_sha256'];return data
 index=np.load(CACHE,mmap_mode='r');forbidden=set(json.loads((R/'excluded-starts.json').read_text())['states'])
 if phase=='holdout':forbidden.update(state_hash(prefix_state(p)) for p in read_prefixes(R/'validation-prefixes.json')['prefixes'])
 rng=random.Random(seed);prefixes=[];seen=set();reject=0
 while len(prefixes)<count:
  s=State();moves=[]
  for _ in range(rng.randint(2,6)):
   _,qs=observe(s,4,'native');m=rng.choice(qs).move if qs else None;moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
  h=state_hash(s)
  if h in seen or h in forbidden or contains(index,h):reject+=1;continue
  seen.add(h);prefixes.append(dict(id=digest(moves),moves=moves,state_hash=h))
 data=dict(format=1,seed=seed,count=count,prefixes=prefixes,sha256=digest(prefixes),rejections=reject,training_union_sha256=json.loads((R/'training-audit.json').read_text())['union_sha256'],pre_separation_pass=True);write(path,data);return data
class Heuristic:
 def __init__(self):self.times=[]
 def act(self,s,rng):
  t=time.perf_counter();obs,qs=observe(s,4,'native')
  if qs:
   scores=heuristic(np.ascontiguousarray(obs[2][:,:12]));m=qs[int(rng.choice(np.flatnonzero(scores>=scores.max()-1e-7)))].move
  else:m=None
  self.times.append(time.perf_counter()-t);return m

def match(checkpoint,opponent,prefixes,path):
 if Path(path).exists():
  d=json.loads(Path(path).read_text());assert d['checkpoint_sha256']==sha(checkpoint) and d['opponent_sha256']==(None if opponent=='heuristic' else sha(opponent)) and d['prefix_sha256']==prefixes['sha256'];return d
 a=PPOAgent(checkpoint=checkpoint,device='cuda',backend='native');b=Heuristic() if opponent=='heuristic' else PPOAgent(checkpoint=opponent,device='cuda',backend='native');games=[];start=time.perf_counter()
 for i,prefix in enumerate(prefixes['prefixes']):
  for seat in (0,1):
   s=prefix_state(prefix);rngs=[random.Random(1759000+i*2+j) for j in (0,1)];moves=[]
   while not s.done:
    m=(a if s.turn==seat else b).act(s,rngs[s.turn]);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   games.append(dict(prefix_id=prefix['id'],seat=seat,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),moves=moves))
 result=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),opponent=str(opponent),opponent_sha256=None if opponent=='heuristic' else sha(opponent),prefix_sha256=prefixes['sha256'],summary=summarize(games),games=games,wall=time.perf_counter()-start,left_ms=1000*np.mean(a.times),right_ms=1000*np.mean(b.times),search=False,all_legal_actions=True);write(path,result);return result

def cluster(matches):
 groups={}
 for seed,result in matches:
  for g in result['games']:groups.setdefault(g['prefix_id'],[]).append(g['difference'])
 assert all(len(v)==6 for v in groups.values())
 dif=np.array([np.mean(v) for v in groups.values()]);points=np.array([np.mean(np.sign(v)*.5+.5) for v in groups.values()]);ix=np.random.default_rng(1759900).integers(len(dif),size=(4000,len(dif)))
 return dict(prefixes=len(dif),games=sum(len(v) for v in groups.values()),difference=float(dif.mean()),difference_ci95=np.quantile(dif[ix].mean(1),[.025,.975]).tolist(),point=float(points.mean()),point_ci95=np.quantile(points[ix].mean(1),[.025,.975]).tolist(),bootstrap_unit='one prefix containing all three seeds and both seats',by_seed={str(s):r['summary'] for s,r in matches})

def validation():
 seed_all(1754000);prefixes=bank('validation',32,17541000);end=json.loads((R/'training-complete.json').read_text())['common_iteration'];steps=[x for x in (1024,2048,3072,4096) if x<=end];selected={};curves=[]
 for seed in SEEDS:
  folder=R/f'seed-{seed}';rank=[]
  for step in steps:
   scores={}
   for label,opp in [('untrained',folder/'inference-0.pt'),('heuristic','heuristic'),('gen2',Path(f'artifacts/ppo-gen2/seed-{seed-1750000}.pt'))]:
    result=match(folder/f'inference-{step}.pt',opp,prefixes,R/f'validation-{seed}-{step}-{label}.json');scores[label]=result['summary']['point_rate']
   criterion=.5*scores['untrained']+.25*scores['heuristic']+.25*scores['gen2'];rank.append((criterion,-step,scores));curves.append(dict(seed=seed,step=step,criterion=criterion,scores=scores));print('VALIDATION',curves[-1],flush=True)
  criterion,negative,scores=max(rank,key=lambda row:(row[0],row[1]));step=-negative;p=folder/f'inference-{step}.pt';selected[str(seed)]=dict(path=str(p),sha256=sha(p),iteration=step,criterion=criterion,scores=scores)
 gate=np.mean([x['scores']['untrained'] for x in selected.values()])>.5 and np.mean([x['scores']['heuristic'] for x in selected.values()])>=.5
 write(R/'selection.json',dict(selected=selected,gate=bool(gate),curves=curves,selection_only='new validation',fixed_before_holdout=True));print('GATE',gate,flush=True)

def holdout():
 seed_all(1756000);lock=json.loads((R/'selection.json').read_text());assert lock['gate'];prefixes=bank('holdout',96,17561000);groups={label:[] for label in ('untrained','heuristic','gen2')}
 for seed in SEEDS:
  chosen=lock['selected'][str(seed)];assert sha(chosen['path'])==chosen['sha256']
  for label,opp in [('untrained',R/f'seed-{seed}'/'inference-0.pt'),('heuristic','heuristic'),('gen2',Path(f'artifacts/ppo-gen2/seed-{seed-1750000}.pt'))]:
   result=match(chosen['path'],opp,prefixes,R/f'holdout-{seed}-{label}.json');groups[label].append((seed,result));print('HOLDOUT',seed,label,result['summary'],flush=True)
 write(R/'holdout-summary.json',{label:cluster(matches) for label,matches in groups.items()})

def audit_eval():
 index=np.load(CACHE,mmap_mode='r');reports=[]
 for phase in ('validation','holdout'):
  if not (R/f'{phase}-prefixes.json').exists():continue
  prefixes=read_prefixes(R/f'{phase}-prefixes.json');mapping={p['id']:p for p in prefixes['prefixes']};assert all(not contains(index,state_hash(prefix_state(p))) for p in prefixes['prefixes'])
  for path in sorted(R.glob(f'{phase}-*.json')):
   d=json.loads(path.read_text())
   if 'games' not in d:continue
   assert sha(d['checkpoint'])==d['checkpoint_sha256'];assert d['opponent_sha256']==(None if d['opponent']=='heuristic' else sha(d['opponent']));assert d['prefix_sha256']==prefixes['sha256'];overlap=set();later=0
   for g in d['games']:
    s=prefix_state(mapping[g['prefix_id']]);assert not contains(index,state_hash(s))
    for row in g['moves']:
     h=state_hash(s)
     if contains(index,h):overlap.add(h);later+=1
     m=Move(*row) if row else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False)
    h=state_hash(s)
    if contains(index,h):overlap.add(h);later+=1
    assert s.done and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat'])
   assert summarize(d['games'])==d['summary'];reports.append(dict(path=str(path),start_overlap=0,later_state_overlap=len(overlap),later_visits=later,games=len(d['games'])))
 write(R/'evaluation-audit.json',dict(pass_=True,rows=reports,all_games_replayed=True,post_visit_overlap_reported=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','holdout','audit']);a=p.parse_args();{'validation':validation,'holdout':holdout,'audit':audit_eval}[a.mode]()
