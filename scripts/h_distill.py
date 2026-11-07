"""Finite frozen-H policy distillation. Production code/models never mutated."""
import sys,copy,json,random,time,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from blokus.engine import State,Move,play,legal_moves
from blokus.ppo_policy import PPOAgent,collate
from blokus.ppo_search_deeper import DeepSearch
from blokus.ppo_runtime import state_hash
R=Path('artifacts/h-policy-distill');SEEDS=(1751401,1751402,1751403)
INDEX=Path('/tmp/ppo-lr64-training.npy')
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw(m):return None if m is None else [m.piece,m.orientation,m.row,m.col]
def state(ms):
 s=State()
 for m in ms:play(s,Move(*m) if m else None,validate=True)
 return s
def contains(idx,h):
 v=np.bytes_(bytes.fromhex(h));i=np.searchsorted(idx,v);return i<len(idx) and idx[i]==v
def exclusions():
 if (R/'excluded.json').exists():return set(json.loads((R/'excluded.json').read_text())['states'])
 forbidden=set();files=[]
 for base in Path('/home/eletim').glob('blokus*'):
  root=base/'artifacts'
  if not root.is_dir() or base.resolve()==Path.cwd().resolve():continue
  for f in sorted(set(root.rglob('*prefix*.json'))|set(root.rglob('*bank.json'))|set(root.rglob('excluded-starts.json'))):
   try:d=json.loads(f.read_text())
   except (ValueError,OSError):continue
   if not isinstance(d,dict):continue
   if isinstance(d.get('states'),list):forbidden.update(h for h in d['states'] if isinstance(h,str))
   for p in d.get('prefixes',[]) if isinstance(d.get('prefixes',[]),list) else []:
    if isinstance(p,dict) and 'moves' in p:
     try:forbidden.add(state_hash(state(p['moves'])))
     except (KeyError,TypeError,ValueError,AssertionError):pass
   files.append(dict(path=str(f),sha256=sha(f)))
 write(R/'excluded.json',dict(states=sorted(forbidden),sources=files));return forbidden

def generate(phase,n):
 idx=np.load(INDEX,mmap_mode='r');forbidden=exclusions();start=time.perf_counter()
 external={'pilot':38100100,'train':38200100,'dev':38300100}[phase]
 for seed in SEEDS:
  out=R/f'{phase}-{seed}.json'
  if out.exists():continue
  p=PPOAgent(seed,'cpu',backend='python');teacher=DeepSearch(p,60000,width=4,max_depth=6);games=[];labels=[]
  for gid in range(n):
   rng=random.Random(external+gid);s=State();moves=[];visited=[];pending=[]
   for ply in range(44):
    if s.done:break
    h=state_hash(s)
    if ply>=6 and h in forbidden:raise RuntimeError('old evaluation state encountered; stop and retain evidence')
    visited.append(h)
    obs,qs=p._observation(s)
    if ply in (12,20,28) and qs and not contains(idx,h):
     greedy=p.act(s,random.Random(0));m=teacher.act(s,random.Random(0));stats=copy.deepcopy(teacher.stats)
     with torch.no_grad():l,_=p.model(*collate([obs],'cpu'));order=l[0].argsort(descending=True).tolist()[:4]
     roots=[qs[i].move for i in order]
     if greedy not in roots:roots[-1]=greedy
     if stats['depth']==6:
      scores=stats['root_scores'];best=max(scores);targets=[raw(roots[i]) for i,v in enumerate(scores) if abs(v-best)<=1e-7]
      pending.append(dict(game=gid,ply=ply,state_hash=h,moves=copy.deepcopy(moves),target=raw(m),targets=targets,greedy=raw(greedy),changed=m!=greedy,stats=stats))
    if not qs:m=None
    elif ply<6:m=rng.choice(qs).move
    elif rng.random()<.2:
     with torch.no_grad():l,_=p.model(*collate([obs],'cpu'));ix=l[0].argsort(descending=True).tolist()[:8]
     m=qs[rng.choice(ix)].move
    else:m=p.act(s,rng)
    play(s,m,validate=True);moves.append(raw(m))
   assert s.done
   games.append(dict(id=gid,moves=moves,visited=visited+[state_hash(s)],scores=[s.score(0),s.score(1)]));labels.extend(pending)
   if gid%16==0:print(phase,seed,gid,len(labels),flush=True)
  write(out,dict(seed=seed,phase=phase,external_seed=external,games=games,labels=labels,teacher_model_sha256=sha(f'artifacts/ppo-gen3/seed-{seed}.pt')))
 report={}
 for seed in SEEDS:
  d=json.loads((R/f'{phase}-{seed}.json').read_text());ls=d['labels'];report[str(seed)]=dict(games=len(d['games']),labels=len(ls),changed=sum(x['changed'] for x in ls),mean_teacher_ms=float(np.mean([x['stats']['elapsed_ms'] for x in ls])),mean_nodes=float(np.mean([x['stats']['nodes'] for x in ls])))
 write(R/f'{phase}-summary.json',dict(wall_seconds=time.perf_counter()-start,by_seed=report));print(report,flush=True)

def setup():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);R.mkdir(exist_ok=True,parents=True)
if __name__=='__main__':
 setup();a=argparse.ArgumentParser();a.add_argument('phase',choices=['pilot','train','dev']);a.add_argument('--games',type=int,default=4);args=a.parse_args();generate(args.phase,args.games)
