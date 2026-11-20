"""Predeclared doubling of A's training data; immutable prior studies."""
import os,copy,gzip,json,random,time,concurrent.futures,argparse
from pathlib import Path
import numpy as np
import torch
import placement_scale as s
import placement_experiment as e
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
from blokus.placement_value import PlacementValue,PlacementEvaluator,mobility
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.value_search import encode
OLD=s.R;R=Path('artifacts/value-placement-6144');SEEDS=e.SEEDS
PAIRS=[('P','F'),('Q','F'),('A','F'),('A','Q'),('A','C'),('C','F')]
def read(p):return json.loads(Path(p).read_text())
def setup():e.setup();R.mkdir(exist_ok=True)
def prepare():
 setup();assert not (R/'exclusions.json').exists()
 ex=set(read(OLD/'exclusions.json')['states'])
 for phase in ('train','dev'):ex.update(read(OLD/f'{phase}-labels.json')['visited'])
 for d,phase in [(OLD/'n384','curve'),(OLD/'n1536','curve'),(OLD/'n3072','validation'),(OLD/'n3072','confirmation'),(OLD/'earlystop','confirmation')]:
  bank=read(d/f'{phase}-bank.json')['prefixes'];ex.update(p['state_hash'] for p in bank)
  for p in d.glob(f'{phase}-??-*.json.gz'):
   for g in json.load(gzip.open(p,'rt'))['games']:
    z=e.state(bank[g['prefix']]['moves']);ex.add(state_hash(z))
    for m in g['moves']:play(z,Move(*m) if m else None,validate=False);ex.add(state_hash(z))
 e.write(R/'exclusions.json',dict(states=sorted(ex)))
 print('PREPARED',len(ex),flush=True)
def worker(args):
 s.R=R
 return s.worker(args)
def generate():
 setup();(R/'games').mkdir(exist_ok=True);tasks=[('train',seed,gid) for seed in SEEDS for gid in range(1024,2048)];start=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:results=list(pool.map(worker,[(i,tasks[i::8]) for i in range(8)]))
 assert len(list((R/'games').glob('*.gz')))==3072
 e.write(R/'generation-time.json',dict(seconds=time.perf_counter()-start,workers=results));print('GENERATED',flush=True)
def features():
 setup();start=time.perf_counter();idx=np.load(e.INDEX,mmap_mode='r');ex=set(read(R/'exclusions.json')['states']);rows=[];labels=[];seen=set();excluded=0
 for pi,p in enumerate(sorted((R/'games').glob('*.gz'))):
  g=json.load(gzip.open(p,'rt'));z=State();local=[]
  for ply,m in enumerate(g['moves']):
   if ply>=8:
    h=state_hash(z);seen.add(h)
    if h in ex or e.contains(idx,h):excluded+=1
    else:
     b,st=encode(z);y=float(np.sign(g['scores'][z.turn]-g['scores'][1-z.turn]));local.append((b[:4],st,mobility(z),y,dict(seed=g['seed'],game=g['id'],ply=ply,turn=z.turn,hash=h,target=y)))
   play(z,Move(*m) if m else None,validate=False)
  assert z.done and g['scores']==[z.score(0),z.score(1)] and local
  for b,st,mob,y,label in local:rows.append((b,st,mob,y,1/len(local),g['id']));labels.append(label)
  if pi%256==0:print('features',pi,flush=True)
 # New generation must not silently reuse old train/dev/evaluation states.
 assert not seen&ex and not any(e.contains(idx,h) for h in seen)
 old=np.load(OLD/'train-features.npz');oldlabels=read(OLD/'train-labels.json')
 arrays={f'x{i}':np.concatenate([old[f'x{i}'],np.stack([r[i] for r in rows]).astype(np.float32)]) for i in range(6)}
 np.savez_compressed(R/'train-features.npz',**arrays)
 e.write(R/'train-labels.json',dict(labels=oldlabels['labels']+labels,visited=sorted(set(oldlabels['visited'])|seen),excluded=excluded,new_visited=sorted(seen)))
 assert not set(read(OLD/'dev-labels.json')['visited'])&set(read(R/'train-labels.json')['visited'])
 e.write(R/'feature-time.json',dict(seconds=time.perf_counter()-start,new_labels=len(labels),all_labels=len(arrays['x0'])));print('FEATURED',len(arrays['x0']),flush=True)
def train():
 setup();assert torch.cuda.is_available();start=time.perf_counter();device='cuda';tr=np.load(R/'train-features.npz');dv=np.load(OLD/'dev-features.npz');x=[torch.tensor(tr[f'x{i}'],device=device) for i in range(5)];v=[torch.tensor(dv[f'x{i}'],device=device) for i in range(5)];x[4]/=x[4].mean()
 for seed in SEEDS:
  p=PPOAgent(seed,device,backend='native');frozen=copy.deepcopy(p.model.state_dict());base=PlacementValue(p.model,'C').to(device)
  with torch.no_grad():old=torch.cat([base(*(a[k:k+2048] for a in x[:3])) for k in range(0,len(x[0]),2048)]).detach()
  for variant in 'AC':
   assert not (R/f'{variant}-{seed}.pt').exists();torch.manual_seed(seed);m=PlacementValue(p.model,variant).to(device);opt=torch.optim.Adam(m.parameters(),lr=1e-4);rng=np.random.default_rng(seed+7821);updates=0;curve=[dict(epoch=0,train_mse=s.mse(m,x),dev_mse=s.mse(m,v))]
   for epoch in range(16):
    order=rng.permutation(len(x[0]))
    for k in range(0,len(order),64):
     ix=order[k:k+64];pred=m(*(a[ix] for a in x[:3]));loss=(x[4][ix]*((pred-x[3][ix]).square()+.1*(pred-old[ix]).square())).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters());opt.step();updates+=1
    curve.append(dict(epoch=epoch+1,train_mse=s.mse(m,x),dev_mse=s.mse(m,v)));print('train',variant,seed,epoch+1,flush=True)
   assert all(torch.equal(t,frozen[k]) for k,t in p.model.state_dict().items())
   torch.save(dict(format='placement-value-v1',variant=variant,seed=seed,model={k:t.detach().cpu() for k,t in m.state_dict().items()},source_sha=e.sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),train_games=6144,epochs=16,updates=updates,data_sha=e.sha(R/'train-features.npz'),dev_sha=e.sha(OLD/'dev-features.npz')),R/f'{variant}-{seed}.pt')
   torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,updates=updates),R/f'resume-{variant}-{seed}.pt');e.write(R/f'learning-{variant}-{seed}.json',dict(curve=curve,updates=updates,train_labels=len(x[0]),dev_labels=len(v[0])))
 e.write(R/'training-time.json',dict(seconds=time.perf_counter()-start));print('TRAINED',flush=True)
def banks():
 ex=set(read(R/'exclusions.json')['states'])|set(read(R/'train-labels.json')['visited']);idx=np.load(e.INDEX,mmap_mode='r')
 for phase,n,base in [('validation',16,701000000),('confirmation',32,702000000)]:
  rng=random.Random(base);bank=[]
  for i in range(n):pr=e.prefix(rng,idx,ex,n=8+i%3*2);ex.add(pr['state_hash']);bank.append(pr)
  e.write(R/f'{phase}-bank.json',dict(prefixes=bank))
 files=[str(p) for p in R.glob('[AC]-*.pt')]+[str(p) for p in R.glob('*bank.json')]+['scripts/placement_more.py','scripts/placement_scale.py','scripts/placement_experiment.py','blokus/placement_value.py','blokus/ppo_search_deeper.py']+[str(OLD/f'n{n}'/f'A-{seed}.pt') for n in (1536,3072) for seed in SEEDS]
 e.write(R/'evaluation-manifest.json',{p:e.sha(p) for p in files})
def agent(name,p,budget=500,depth=6):
 if name=='F':return DeepSearch(p,budget,max_depth=depth)
 src=OLD/f'n{1536 if name=="P" else 3072}'/f'A-{p.model_seed}.pt' if name in 'PQ' else R/f'{name}-{p.model_seed}.pt'
 ck=torch.load(src,map_location='cpu',weights_only=True);assert ck['source_sha']==e.sha(f'artifacts/ppo-gen3/seed-{p.model_seed}.pt');m=PlacementValue(p.model,ck['variant']);m.load_state_dict(ck['model']);return DeepSearch(p,budget,max_depth=depth,value_evaluator=PlacementEvaluator(m))
def aggregate(phase,pair):
 e.R=R;out=e.aggregate(phase,pair)
 if phase=='confirmation':
  rows=[g for seed in SEEDS for g in json.load(gzip.open(R/f'{phase}-{pair}-{seed}.json.gz','rt'))['games']];groups=sorted({g['prefix'] for g in rows});pt=np.array([np.mean([g['point'] for g in rows if g['prefix']==i]) for i in groups]);df=np.array([np.mean([g['difference'] for g in rows if g['prefix']==i]) for i in groups]);ix=np.random.default_rng(771).integers(len(groups),size=(20000,len(groups)));out['point_adjusted_ci']=np.quantile(pt[ix].mean(1),[.05/6,1-.05/6]).tolist();out['difference_adjusted_ci']=np.quantile(df[ix].mean(1),[.05/6,1-.05/6]).tolist()
 return out
def evaluate(phase):
 e.R=R;e.agent=agent
 if phase=='confirmation':assert read(R/'selection.json')['passed']
 pairs=PAIRS if phase=='validation' else [('A','F'),('A','Q'),('A','C')];e.evaluate(phase,pairs);out={a+b:aggregate(phase,a+b) for a,b in pairs};e.write(R/f'{phase}-summary.json',out)
 if phase=='validation':passed=all(2*out[p]['wins']+out[p]['draws']>out[p]['games'] and out[p]['difference']>0 for p in ('AF','AQ')) and 2*out['AC']['wins']+out['AC']['draws']>=out['AC']['games'] and all(out[p]['stats']['A']['over_550']<=.01 for p in ('AF','AQ','AC'))
 else:passed=all(v['point_adjusted_ci'][0]>.5+1e-9 and v['difference']>0 and v['stats']['A']['over_550']<=.01 for v in out.values())
 e.write(R/('selection.json' if phase=='validation' else 'decision.json'),dict(passed=passed,results=out));print('DECISION',phase,passed,{k:v['point'] for k,v in out.items()},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','generate','features','train','banks','validation','confirmation']);a=ap.parse_args()
 if a.command in ('validation','confirmation'):evaluate(a.command)
 else:globals()[a.command]()
