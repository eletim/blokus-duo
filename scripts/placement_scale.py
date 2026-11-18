"""Predeclared data-scaling extension; original small pilot remains immutable."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, concurrent.futures, gzip, json, random, time, copy, hashlib
from pathlib import Path
import numpy as np
import torch
import placement_experiment as e
from blokus.engine import State,Move,play
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.ppo_runtime import state_hash
from blokus.placement_value import PlacementValue,mobility
from blokus.value_search import encode
R=Path('artifacts/value-placement-scale')
STAGES=(384,1536,3072)
SEEDS=e.SEEDS

def setup():
 e.setup();R.mkdir(parents=True,exist_ok=True)

def exclusions():
 return set(json.loads((R/'exclusions.json').read_text())['states'])

def prepare():
 setup();assert not (R/'exclusions.json').exists()
 ex=e.forbidden();old=e.R
 for phase in ('train','dev'):
  ex.update(json.loads((old/f'{phase}-labels.json').read_text())['visited'])
 for phase in ('validation','fixed','confirmation'):
  bank=json.loads((old/f'{phase}-bank.json').read_text())['prefixes']
  ex.update(p['state_hash'] for p in bank)
  for path in old.glob(f'{phase}-??-*.json.gz'):
   for g in json.load(gzip.open(path,'rt'))['games']:
    s=e.state(bank[g['prefix']]['moves']);ex.add(state_hash(s))
    for m in g['moves']:
     play(s,Move(*m) if m else None,validate=False);ex.add(state_hash(s))
 e.write(R/'exclusions.json',dict(states=sorted(ex),old_report_sha=e.sha(old/'REPORT.md')))
 e.write(R/'plan.json',dict(train_games=list(STAGES),dev_games=192,epochs=16,batch_size=64,lr=1e-4,regularization=.1,seeds=list(SEEDS),variants=list('ABC'),teacher='frozen gen3 H/500ms width4 depth6',generator_workers=8,validation_prefixes=16,confirmation_prefixes=32,curve_prefixes=6))
 print('PREPARED',len(ex),flush=True)

def worker(args):
 worker_id,tasks=args
 os.sched_setaffinity(0,{worker_id})
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 idx=np.load(e.INDEX,mmap_mode='r');ex=exclusions();agents={};start=time.perf_counter();done=0
 for phase,seed,gid in tasks:
  path=R/'games'/f'{phase}-{seed}-{gid:04d}.json.gz'
  if path.exists():
   g=json.load(gzip.open(path,'rt'));assert (g['phase'],g['seed'],g['id'])==(phase,seed,gid);continue
  if seed not in agents:agents[seed]=DeepSearch(PPOAgent(seed,'cpu',backend='native'),500)
  a=agents[seed];rng=random.Random((691000000 if phase=='train' else 692000000)+seed*10000+gid)
  pr=e.prefix(rng,idx,ex,n=8);s=e.state(pr['moves']);ms=list(pr['moves']);stats=[]
  while not s.done:
   m=a.act(s,rng);stats.append({k:a.stats[k] for k in ('depth','elapsed_ms','forced_pass','fallback')});play(s,m,validate=True);ms.append(e.raw(m))
  g=dict(phase=phase,seed=seed,id=gid,prefix_length=8,moves=ms,scores=[s.score(0),s.score(1)],stats=stats,source_sha=e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt'))
  temp=path.with_suffix('.tmp')
  with gzip.open(temp,'wt') as f:json.dump(g,f)
  temp.replace(path);done+=1
  if done%8==0:print('generate worker',worker_id,'games',done,'seconds',round(time.perf_counter()-start),flush=True)
 return dict(worker=worker_id,new_games=done,seconds=time.perf_counter()-start)

def generate():
 setup();(R/'games').mkdir(exist_ok=True)
 idx=np.load(e.INDEX,mmap_mode='r');assert hashlib.sha256(idx.tobytes()).hexdigest()=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c'
 tasks=[(phase,seed,gid) for phase,n in [('train',1024),('dev',64)] for seed in SEEDS for gid in range(n)]
 start=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:results=list(pool.map(worker,[(i,tasks[i::8]) for i in range(8)]))
 assert len(list((R/'games').glob('*.json.gz')))==3264
 e.write(R/'generation-time.json',dict(seconds=time.perf_counter()-start,workers=results));print('GENERATION COMPLETE',flush=True)

def features():
 setup();idx=np.load(e.INDEX,mmap_mode='r');ex=exclusions();sets={};start=time.perf_counter()
 for phase in ('train','dev'):
  rows=[];visited=set();rejected=0;labels=[]
  for pi,path in enumerate(sorted((R/'games').glob(f'{phase}-*.json.gz'))):
   g=json.load(gzip.open(path,'rt'));s=State();local=[]
   for ply,m in enumerate(g['moves']):
    if ply>=g['prefix_length']:
     h=state_hash(s);visited.add(h)
     if h in ex or e.contains(idx,h):rejected+=1
     else:
      b,st=encode(s);y=float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn]));local.append((b[:4],st,mobility(s),y,dict(seed=g['seed'],game=g['id'],ply=ply,turn=s.turn,hash=h,target=y)))
    play(s,Move(*m) if m else None,validate=False)
   assert s.done and [s.score(0),s.score(1)]==g['scores'] and local
   for b,st,mob,y,label in local:rows.append((b,st,mob,y,1/len(local),g['id']));labels.append(label)
   if pi%128==0:print('features',phase,pi,flush=True)
  np.savez_compressed(R/f'{phase}-features.npz',**{f'x{i}':np.stack([r[i] for r in rows]).astype(np.float32) for i in range(6)})
  e.write(R/f'{phase}-labels.json',dict(labels=labels,visited=sorted(visited),excluded=rejected));sets[phase]=visited
 assert not sets['train']&sets['dev']
 e.write(R/'feature-time.json',dict(seconds=time.perf_counter()-start));print('FEATURES COMPLETE',flush=True)

def mse(model,arrays,original=None):
 total=0.;weight=0.
 with torch.no_grad():
  for offset in range(0,len(arrays[0]),2048):
   sl=slice(offset,offset+2048);pred=model(*(a[sl] for a in arrays[:3]));w=arrays[4][sl];total+=float(((pred-arrays[3][sl]).square()*w).sum());weight+=float(w.sum())
 return total/weight

def train():
 setup();assert torch.cuda.is_available();device='cuda';start=time.perf_counter()
 train=np.load(R/'train-features.npz');dev=np.load(R/'dev-features.npz');v=[torch.tensor(dev[f'x{i}'],device=device) for i in range(5)]
 data_shas={p.name:e.sha(p) for p in R.glob('*-features.npz')}
 for size in STAGES:
  d=R/f'n{size}';d.mkdir(exist_ok=True);mask=train['x5']<size//3;x=[torch.tensor(train[f'x{i}'][mask],device=device) for i in range(5)];x[4]/=x[4].mean()
  for seed in SEEDS:
   p=PPOAgent(seed,device,backend='native');frozen=copy.deepcopy(p.model.state_dict());base=PlacementValue(p.model,'C').to(device)
   with torch.no_grad():old=torch.cat([base(*(a[k:k+2048] for a in x[:3])) for k in range(0,len(x[0]),2048)]).detach()
   for variant in 'ABC':
    assert not (d/f'{variant}-{seed}.pt').exists()
    torch.manual_seed(seed);m=PlacementValue(p.model,variant).to(device);opt=torch.optim.Adam(m.parameters(),lr=1e-4);rng=np.random.default_rng(seed+7821);updates=0;curve=[dict(epoch=0,train_mse=mse(m,x),dev_mse=mse(m,v))];t=time.perf_counter()
    for epoch in range(16):
     order=rng.permutation(len(x[0]))
     for k in range(0,len(order),64):
      ix=order[k:k+64];pred=m(*(a[ix] for a in x[:3]));loss=(x[4][ix]*((pred-x[3][ix]).square()+.1*(pred-old[ix]).square())).mean();assert torch.isfinite(loss)
      opt.zero_grad();loss.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters());opt.step();updates+=1
     curve.append(dict(epoch=epoch+1,train_mse=mse(m,x),dev_mse=mse(m,v)))
    assert all(torch.equal(t,frozen[k]) for k,t in p.model.state_dict().items())
    torch.save(dict(format='placement-value-v1',variant=variant,seed=seed,model={k:t.detach().cpu() for k,t in m.state_dict().items()},source_sha=e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),data_shas=data_shas,train_games=size,epochs=16,updates=updates),d/f'{variant}-{seed}.pt')
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,updates=updates),d/f'resume-{variant}-{seed}.pt')
    e.write(d/f'learning-{variant}-{seed}.json',dict(train_games=size,train_labels=len(x[0]),dev_labels=len(v[0]),updates=updates,curve=curve,seconds=time.perf_counter()-t))
    print('TRAIN',size,variant,seed,updates,curve[-1],flush=True)
 e.write(R/'training-time.json',dict(seconds=time.perf_counter()-start));print('TRAINING COMPLETE',flush=True)

def banks():
 setup();idx=np.load(e.INDEX,mmap_mode='r');ex=exclusions()
 for phase in ('train','dev'):ex.update(json.loads((R/f'{phase}-labels.json').read_text())['visited'])
 for phase,n,base in [('curve',6,693000000),('validation',16,694000000),('confirmation',32,695000000)]:
  rng=random.Random(base);bank=[]
  for i in range(n):
   pr=e.prefix(rng,idx,ex,n=8+(i%3)*2);ex.add(pr['state_hash']);bank.append(pr)
  for size in STAGES:e.write(R/f'n{size}'/f'{phase}-bank.json',dict(prefixes=bank))
 files=['scripts/placement_scale.py','scripts/placement_experiment.py','blokus/placement_value.py','blokus/ppo_search_deeper.py']+[str(p) for p in R.glob('n*/[ABC]-*.pt')]+[str(p) for p in R.glob('n*/*-bank.json')]
 manifest={p:e.sha(p) for p in files}
 for size in STAGES:e.write(R/f'n{size}'/'evaluation-manifest.json',manifest)
 print('BANKS AND MODELS FROZEN',flush=True)

def evaluate(size,phase):
 e.R=R/f'n{size}'
 pairs=[('A','F'),('B','F'),('C','F')] if phase=='curve' else [('A','F'),('B','F'),('C','F'),('A','C'),('B','C')]
 if phase=='confirmation':
  choice=json.loads((R/'selection.json').read_text())['selected'];assert choice in ('A','B');pairs=[(choice,'F'),(choice,'C')]
 e.evaluate(phase,pairs)
 e.write(e.R/f'{phase}-summary.json',{a+b:e.aggregate(phase,a+b) for a,b in pairs})

def select():
 e.R=R/'n3072';out={p:e.aggregate('validation',p) for p in ('AF','BF','CF','AC','BC')};eligible=[]
 for a in 'AB':
  f=out[a+'F'];c=out[a+'C']
  if f['point']>.5 and f['difference']>0 and c['point']>=.5 and f['stats'][a]['over_550']<=.01:eligible.append(a)
 chosen=max(eligible,key=lambda a:(out[a+'F']['point'],a=='A')) if eligible else None
 e.write(R/'selection.json',dict(results=out,eligible=eligible,selected=chosen));print('SELECTION',chosen,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','generate','features','train','banks','curve','validation','select','confirmation']);ap.add_argument('--size',type=int,choices=STAGES,default=3072);args=ap.parse_args()
 if args.command in ('curve','validation','confirmation'):evaluate(args.size,args.command)
 else:globals()[args.command]()
