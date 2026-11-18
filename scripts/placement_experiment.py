"""Finite A/B/C value experiment. Run from repository root; no production writes."""
import argparse,copy,gzip,hashlib,json,random,time,glob
from pathlib import Path
import numpy as np
import torch
from blokus.engine import State,Move,play,legal_moves
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.ppo_runtime import state_hash
from blokus.placement_value import PlacementValue,PlacementEvaluator,mobility
from blokus.value_search import encode
R=Path('artifacts/value-placement-ab');SEEDS=(1751401,1751402,1751403);INDEX='/tmp/ppo-lr64-training.npy'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def raw(m):return [m.piece,m.orientation,m.row,m.col] if m else None
def state(ms):
 s=State()
 for m in ms:play(s,Move(*m) if m else None,validate=False)
 return s
def contains(idx,h):
 b=np.bytes_(bytes.fromhex(h));k=np.searchsorted(idx,b);return k<len(idx) and idx[k]==b
def setup():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);R.mkdir(parents=True,exist_ok=True)
def forbidden():
 p=R/'historical-starts.json'
 if p.exists():return set(json.loads(p.read_text())['states'])
 files=set();out=set();sources=[]
 for pattern in ('*bank.json','*prefix*.json','excluded-starts.json'):
  files.update(glob.glob('/home/eletim/blokus*/artifacts/**/'+pattern,recursive=True))
 def collect(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k=='state_hash' and isinstance(v,str) and len(v)==64:out.add(v)
    elif k=='states' and isinstance(v,list):out.update(h for h in v if isinstance(h,str) and len(h)==64)
    else:collect(v)
  elif isinstance(x,list):
   for v in x:collect(v)
 for name in sorted(files):
  if 'value-placement-ab' in name:continue
  try:d=json.loads(Path(name).read_text())
  except (ValueError,OSError):continue
  collect(d);sources.append(dict(path=name,sha256=sha(name)))
 write(p,dict(states=sorted(out),sources=sources));return out

def prefix(rng,idx,excluded,n=8):
 for attempt in range(1000):
  s=State();ms=[]
  for _ in range(n):
   qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=False);ms.append(raw(m))
  h=state_hash(s)
  if not contains(idx,h) and h not in excluded:return dict(moves=ms,state_hash=h)
 raise RuntimeError('No separated prefix')

def generate():
 setup();idx=np.load(INDEX,mmap_mode='r');ex=forbidden();start=time.perf_counter()
 assert hashlib.sha256(idx.tobytes()).hexdigest()=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c'
 for seed in SEEDS:
  p=PPOAgent(seed,'cpu',backend='native');agent=DeepSearch(p,500)
  for phase,n,base in [('train',32,58100000),('dev',8,58200000)]:
   path=R/f'{phase}-{seed}.json.gz'
   if path.exists():raise RuntimeError('generation already exists')
   games=[]
   for gid in range(n):
    rng=random.Random(base+seed*101+gid);pr=prefix(rng,idx,ex);s=state(pr['moves']);moves=list(pr['moves']);stats=[]
    while not s.done:
     m=agent.act(s,rng);stats.append(copy.deepcopy(agent.stats));play(s,m,validate=True);moves.append(raw(m))
    games.append(dict(id=gid,prefix_length=len(pr['moves']),moves=moves,scores=[s.score(0),s.score(1)],stats=stats))
    print('generate',seed,phase,gid+1,'seconds',round(time.perf_counter()-start),flush=True)
   with gzip.open(path,'wt') as f:json.dump(dict(seed=seed,phase=phase,source_sha=sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),games=games),f)
 write(R/'generation-time.json',dict(seconds=time.perf_counter()-start))

def data(phase):
 idx=np.load(INDEX,mmap_mode='r');ex=forbidden();rows=[];seen=set();excluded=0
 for seed in SEEDS:
  d=json.load(gzip.open(R/f'{phase}-{seed}.json.gz','rt'))
  for g in d['games']:
   s=State();local=[]
   for ply,m in enumerate(g['moves']):
    if ply>=g['prefix_length']:
     h=state_hash(s);seen.add(h)
     if h in ex or contains(idx,h):excluded+=1
     else:
      b,st=encode(s);y=float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn]));local.append((b[:4],st,mobility(s),y,dict(seed=seed,game=g['id'],ply=ply,turn=s.turn,hash=h)))
    play(s,Move(*m) if m else None,validate=False)
   assert local and s.done and g['scores']==[s.score(0),s.score(1)]
   for b,st,extra,y,label in local:rows.append((b,st,extra,y,1/len(local),label))
 write(R/f'{phase}-labels.json',dict(labels=[r[-1] for r in rows],excluded=excluded,visited=sorted(seen)))
 arrays=[np.stack([r[i] for r in rows]).astype(np.float32) for i in range(5)]
 return arrays,seen

def train():
 setup();start=time.perf_counter();tr,ts=data('train');dv,ds=data('dev');assert not ts&ds
 device='cuda' if torch.cuda.is_available() else 'cpu';x=[torch.tensor(a,device=device) for a in tr];v=[torch.tensor(a,device=device) for a in dv];x[4]/=x[4].mean()
 for seed in SEEDS:
  p=PPOAgent(seed,device,backend='native');original=copy.deepcopy(p.model.state_dict());base=PlacementValue(p.model,'C').to(device)
  with torch.no_grad():old=base(*x[:3]).detach();olddev=base(*v[:3]).detach()
  for variant in 'ABC':
   torch.manual_seed(seed);m=PlacementValue(p.model,variant).to(device);opt=torch.optim.Adam(m.parameters(),lr=1e-4);rng=np.random.default_rng(seed+7821);curve=[];updates=0
   for epoch in range(16):
    order=rng.permutation(len(x[0]))
    for offset in range(0,len(order),64):
     ix=order[offset:offset+64];pred=m(*(a[ix] for a in x[:3]));loss=(x[4][ix]*((pred-x[3][ix])**2+.1*(pred-old[ix])**2)).mean();assert torch.isfinite(loss)
     opt.zero_grad();loss.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters());opt.step();updates+=1
    with torch.no_grad():dev=float(((m(*v[:3])-v[3]).square()*v[4]).sum()/v[4].sum())
    curve.append(dict(epoch=epoch+1,dev_mse=dev))
   assert all(torch.equal(t,original[k]) for k,t in p.model.state_dict().items())
   torch.save(dict(format='placement-value-v1',variant=variant,seed=seed,model={k:t.detach().cpu() for k,t in m.state_dict().items()},source_sha=sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),train_sha=[sha(R/f'train-{i}.json.gz') for i in SEEDS]),R/f'{variant}-{seed}.pt')
   torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,updates=updates),R/f'resume-{variant}-{seed}.pt')
   write(R/f'learning-{variant}-{seed}.json',dict(curve=curve,updates=updates,train_labels=len(x[0]),dev_labels=len(v[0]),before_dev_mse=float(((olddev-v[3]).square()*v[4]).sum()/v[4].sum()),parameters=sum(t.numel() for t in m.parameters())))
   print('trained',variant,seed,updates,curve[-1],flush=True)
 write(R/'training-time.json',dict(seconds=time.perf_counter()-start,device=device))
 # Banks fixed before viewing any playing-strength result.
 ex=forbidden()|ts|ds;idx=np.load(INDEX,mmap_mode='r')
 for phase,n,base in [('validation',12,58300000),('confirmation',32,58400000),('fixed',4,58500000)]:
  rng=random.Random(base);bank=[]
  for i in range(n):
   pr=prefix(rng,idx,ex,n=8+(i%3)*2);ex.add(pr['state_hash']);bank.append(pr)
  write(R/f'{phase}-bank.json',dict(prefixes=bank))
 files=['blokus/placement_value.py','blokus/ppo_search_deeper.py','scripts/placement_experiment.py']+[str(p) for p in R.glob('[ABC]-*.pt')]+[str(p) for p in R.glob('*-bank.json')]
 write(R/'evaluation-manifest.json',{name:sha(name) for name in files})

def agent(name,p,budget=500,depth=6):
 if name=='F':return DeepSearch(p,budget,max_depth=depth)
 ck=torch.load(R/f'{name}-{p.model_seed}.pt',map_location='cpu',weights_only=True);assert ck['source_sha']==sha(f'artifacts/ppo-gen3/seed-{p.model_seed}.pt')
 m=PlacementValue(p.model,name);m.load_state_dict(ck['model']);return DeepSearch(p,budget,max_depth=depth,value_evaluator=PlacementEvaluator(m))

def evaluate(phase,pairs):
 setup();start=time.perf_counter()
 for name,h in json.loads((R/'evaluation-manifest.json').read_text()).items():assert sha(name)==h,name
 bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];budget=60000 if phase=='fixed' else 500;depth=4 if phase=='fixed' else 6
 for left,right in pairs:
  for seed in SEEDS:
   path=R/f'{phase}-{left}{right}-{seed}.json.gz'
   if path.exists():raise RuntimeError('evaluation already exists')
   p=PPOAgent(seed,'cpu',backend='native');aa=agent(left,p,budget,depth);bb=agent(right,p,budget,depth);games=[]
   # Warm up each evaluator, outside timing. All timed games run sequentially.
   for a in (aa,bb):a.act(state(bank[0]['moves']),random.Random(2))
   for pi,pr in enumerate(bank):
    for seat in (0,1):
     s=state(pr['moves']);moves=[];stats=[];rng=random.Random(seed*100+pi*2+seat)
     while not s.done:
      a=aa if s.turn==seat else bb;m=a.act(s,rng);stats.append(dict(player=s.turn,method=left if s.turn==seat else right,**copy.deepcopy(a.stats)))
      if phase=='fixed' and not a.stats['forced_pass']:assert a.stats['depth']==4
      play(s,m,validate=True);moves.append(raw(m))
     scores=[s.score(0),s.score(1)];diff=scores[seat]-scores[1-seat];games.append(dict(prefix=pi,seat=seat,moves=moves,scores=scores,difference=diff,point=(diff>0)+.5*(diff==0),stats=stats))
     print('eval',phase,left+right,seed,pi,seat,'seconds',round(time.perf_counter()-start),flush=True)
   with gzip.open(path,'wt') as f:json.dump(dict(left=left,right=right,seed=seed,budget_ms=budget,max_depth=depth,games=games),f)
 write(R/f'{phase}-time.json',dict(seconds=time.perf_counter()-start))

def aggregate(phase,pair):
 rows=[]
 for seed in SEEDS:
  d=json.load(gzip.open(R/f'{phase}-{pair}-{seed}.json.gz','rt'));rows.extend(dict(seed=seed,**g) for g in d['games'])
 groups=sorted(set(g['prefix'] for g in rows));pts=np.array([np.mean([g['point'] for g in rows if g['prefix']==i]) for i in groups]);diff=np.array([np.mean([g['difference'] for g in rows if g['prefix']==i]) for i in groups]);rng=np.random.default_rng(771);ix=rng.integers(len(groups),size=(20000,len(groups)))
 ci=lambda x:np.quantile(x[ix].mean(1),[.0125,.9875]).tolist()
 stats={}
 for name in pair:
  ss=[s for g in rows for s in g['stats'] if s['method']==name and not s['forced_pass']];t=np.array([s['elapsed_ms'] for s in ss]);stats[name]=dict(nonpass=len(ss),mean_ms=float(t.mean()),max_ms=float(t.max()),over_550=float((t>550).mean()),depth6=float(np.mean([s['depth']==6 for s in ss])),fallback=sum(s['fallback'] for s in ss))
 return dict(games=len(rows),wins=sum(g['difference']>0 for g in rows),draws=sum(g['difference']==0 for g in rows),losses=sum(g['difference']<0 for g in rows),point=float(pts.mean()),point_adjusted_ci=ci(pts),difference=float(diff.mean()),difference_adjusted_ci=ci(diff),by_seed={str(s):float(np.mean([g['point'] for g in rows if g['seed']==s])) for s in SEEDS},stats=stats)

def select():
 out={p:aggregate('validation',p) for p in ('AF','BF','CF','AC','BC')};eligible=[]
 for name in 'AB':
  f=out[name+'F'];c=out[name+'C'];s=f['stats'][name]
  if f['point']>=.55 and f['difference']>0 and c['point']>=.5 and sum(v>=.5 for v in f['by_seed'].values())>=2 and s['over_550']<=.01:eligible.append(name)
 selected=max(eligible,key=lambda a:out[a+'F']['point']) if eligible else None
 write(R/'selection.json',dict(results=out,eligible=eligible,selected=selected));print(json.dumps(dict(selected=selected,points={k:v['point'] for k,v in out.items()})),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['generate','train','validation','fixed','select','confirmation']);args=ap.parse_args()
 if args.command=='generate':generate()
 elif args.command=='train':train()
 elif args.command in ('validation','fixed'):evaluate(args.command,[('A','F'),('B','F'),('C','F'),('A','C'),('B','C')])
 elif args.command=='select':select()
 else:
  selected=json.loads((R/'selection.json').read_text())['selected'];assert selected in 'AB';evaluate('confirmation',[(selected,'F'),(selected,'C')])
