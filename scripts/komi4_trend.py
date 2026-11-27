"""Five-generation, bounded local komi4 campaign; atomic games and checkpoints."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,fcntl,gzip,hashlib,json,random,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from blokus.engine import State,Move,play,legal_moves,SIZES
from blokus.gen4 import load
from blokus.ppo_policy import collate
from blokus.margin_search import MarginSearch,key
from blokus.margin_network import MarginNet,NetworkEvaluator,inputs,terminal_margin
from blokus.margin_value import MarginValue
R=Path('artifacts/komi4-trend');PRIOR=Path('artifacts/komi4-scale/models');OLD=Path('artifacts/komi4-scale');SEEDS=[1751401,1751402,1751403]
R.mkdir(exist_ok=True)
def read(p):
 p=Path(p)
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 if p.suffix=='.gz':
  with gzip.open(tmp,'wt') as f:json.dump(x,f,separators=(',',':'))
 else:tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2))
 tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def event(stage,**kw):
 x=dict(at=time.time(),stage=stage,komi=4,**kw)
 with (R/'operations.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 write(R/'status.json',x);print(json.dumps(x),flush=True)
def raw(m):return None if m is None else [m.piece,m.orientation,m.row,m.col]
def state_hash(s):return hashlib.sha256(repr(key(s)).encode()).hexdigest()
def baseline(seed):return str(PRIOR/f'g5-{seed}.pt')
def task(split,i,generation=0,opp=None,seat=0,seed=None):
 seed=seed or SEEDS[i%3]
 group=f'{split}-{i:05}' if split!='train' else f'train-g{generation}-{i:05}'
 return dict(id=f'{split}-g{generation}-{i:05}-{opp or "mix"}-{seat}-{seed}',split=split,generation=generation,index=i,group=group,seed=seed,opponent=opp or ['self','self','gen4','prior','ridge'][i%5],seat=seat,prefix_seed=728000000+{'train':1000000,'validation':2000000,'test':3000000,'benchmark':4000000,'final':5000000,'standard':6000000}[split]+(generation*10000 if split=='train' else 0)+i,prefix_plies=0 if split=='standard' else [8,10,12][i%3],opening='standard' if split=='standard' else 'uniform-legal' if i%4==0 else 'policy-top16-uniform',budget_ms=500 if split in ('benchmark','final','standard') else 150,max_depth=6 if split in ('benchmark','final','standard') else 4,width=4)
def prepare():
 if (R/'plan.json').exists():return
 tasks=[]
 for g in range(6,16):tasks += [task('train',i,g,seat=i%2) for i in range(2200)]
 for split in ['validation','test']:tasks += [task(split,i,opp=['gen4','prior','self','ridge'][i%4],seat=i%2) for i in range(512)]
 for g in range(5,16):
  for i in range(128):
   for seed in SEEDS:
    for opp in ['gen4','prior']:
     for seat in [0,1]:tasks.append(task('benchmark',i,g,opp,seat,seed))
 for split,n in [('final',128),('standard',1)]:
  for i in range(n):
   for seed in SEEDS:
    for opp in ['gen4','prior']:
     for seat in [0,1]:tasks.append(task(split,i,0,opp,seat,seed))
 write(R/'tasks.json',tasks)
 write(R/'plan.json',dict(created_at=time.time(),komi=4,generations=list(range(6,16)),baseline_generation=5,additional_unique_train_target=22000,cumulative_unique_train_target=33000,model_seeds=SEEDS,workers=8,max_wall_seconds=86400,max_epochs=50,min_epochs=8,patience=6,batch_size=512,learning_rate=.0003,selection='per seed minimum NEW validation MAE over baseline g5 and g6..g15; ties retain earliest. Benchmark NEVER selects models. Final held-out used once after selection freeze.',benchmark='same 128 prefixes x 3 seeds x 2 seats x 2 fixed opponents =1536 games per generation (768/opponent,128 independent opening groups). g5 and g6..15 all identical assignments. Repeated progress benchmark, not unknown test.',opening='prefix policy always frozen seed1751401 for ALL model seeds; 25% uniform legal and75% top16-uniform,8/10/12 plies. Exact prefix hashes frozen before training.',final='distinct fixed128 prefixes x3seeds x2seats x2opponents=1536 plus12 standard-start games; one evaluation after freeze',training='2200 new games/generation:40% current selfplay,20% gen4,20% frozen g5,20% ridge. Current generation half loss mass, previous unique new data half; full shuffled epochs.',timing='epoch seconds measure training plus validation wall time, not pure GPU time. Generation/evaluation stage wall seconds separately recorded.',tasks_sha256=sha(R/'tasks.json'),baseline_models={str(s):sha(baseline(s)) for s in SEEDS},stop='24h, nonfinite or resource anomaly; checkpoint before reporting; never indefinite'))
 event('plan-frozen',additional_train_target=22000,benchmark_games=16896,final_games=1548)

_PAIRS={};_VALUES={}
def init_worker():
 torch.set_num_threads(1)
 identity=mp.current_process()._identity;cpu=4+((identity[-1]-1)%8 if identity else 0);os.sched_setaffinity(0,{cpu});os.nice(5)
def pair(seed):
 if seed not in _PAIRS:_PAIRS[seed]=load(seed,'native')
 return _PAIRS[seed]
def net(path):
 if path not in _VALUES:_VALUES[path]=NetworkEvaluator(path)
 return _VALUES[path]
class Gen4Komi:
 def __init__(self,v):self.v=v
 def __call__(self,s):
  if s.done:
   d=terminal_margin(s);return float((d>0)-(d<0))
  return self.v(s)
def prefix(t,policy):
 policy=pair(SEEDS[0])[0]
 s=State();moves=[];rng=random.Random(t['prefix_seed'])
 for _ in range(t['prefix_plies']):
  if t['opening']=='uniform-legal':
   qs=legal_moves(s);m=rng.choice(qs).move if qs else None
  else:
   obs,qs=policy._observation(s)
   if qs:
    with torch.inference_mode():logits,_=policy.model(*collate([obs],'cpu'));order=logits[0].argsort(descending=True).tolist()
    m=qs[rng.choice(order[:16])].move
   else:m=None
  play(s,m,validate=False);moves.append(raw(m))
 return s,moves

def game(job):
 t,modelpath=job;path=R/'games'/(t['id']+'.json.gz');mh=sha(modelpath)
 if path.exists():
  d=read(path);assert d['task']==t and d['model_sha256']==mh;return dict(id=t['id'],seconds=d['seconds'],existing=True)
 policy,v=pair(t['seed']);value=net(modelpath);rv=MarginValue('artifacts/margin-pilot/model.json',4)
 other={'self':value,'gen4':Gen4Komi(v),'ridge':rv,'prior':net(baseline(t['seed']))}[t['opponent']]
 agents=[MarginSearch(policy,t['budget_ms'],t['width'],t['max_depth'],value_evaluator=e) for e in [value,other]]
 s,moves=prefix(t,policy);prefix_moves=copy.deepcopy(moves);start=time.monotonic();stats=[];predictions=[];rngs=[random.Random(t['prefix_seed']+999+p) for p in [0,1]]
 while not s.done:
  p=s.turn;agent=agents[0 if p==t['seat'] else 1]
  # Store prospective root prediction independently of chosen search value.
  if t['split']!='train':predictions.append(dict(ply=len(moves),player=p,adjusted=value(s)))
  m=agent.act(s,rngs[p]);st=copy.deepcopy(agent.stats)
  stats.append(dict(player=p,agent='candidate' if p==t['seat'] else t['opponent'],**st));play(s,m,validate=True);moves.append(raw(m))
 scores=[s.score(p) for p in [0,1]];remaining=[sum(SIZES[i] for i in range(21) if s.remaining[p]>>i&1) for p in [0,1]]
 d=dict(task=t,model_sha256=mh,moves=moves,prefix=prefix_moves,stats=stats,predictions=predictions,scores=scores,remaining=remaining,adjusted_first_margin=scores[0]-scores[1]-4,trace_sha256=hashlib.sha256(json.dumps(moves).encode()).hexdigest(),seconds=time.monotonic()-start)
 write(path,d);return dict(id=t['id'],seconds=d['seconds'],existing=False)

def generation_models(g):
 return {str(s):baseline(s) if g==6 else str(R/f'models/g{g-1}-{s}.pt') for s in SEEDS}
def run_games(tasks,models,label):
 start=time.monotonic();done=0;seconds=[];existing=0
 with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
  jobs=[pool.submit(game,(t,models[str(t['seed'])])) for t in tasks]
  for f in as_completed(jobs):
   x=f.result();done+=1;seconds.append(x['seconds']);existing+=x['existing']
   if done%32==0 or done==len(tasks):event(label,completed=done,total=len(tasks),generation=tasks[0]['generation'],seconds=time.monotonic()-start,existing=existing)
 result=dict(games=len(tasks),new=len(tasks)-existing,wall_seconds=time.monotonic()-start,worker_game_seconds=sum(seconds),workers=8)

 timing_path=R/(label+'-timing.json')
 if result['new'] or not timing_path.exists():write(timing_path,result)
 return result

def features_one(path):
 d=read(path);s=State();b=[];st=[];ex=[];y=[];keys=[];plies=[];ridge=[];rv=MarginValue('artifacts/margin-pilot/model.json',4)
 for ply,m in enumerate(d['moves']):
  if ply>=len(d['prefix']) and not s.done:
   keys.append(state_hash(s));a,c,e=inputs(s);b.append(a);st.append(c);ex.append(e);plies.append(ply)
   delta=d['adjusted_first_margin'];y.append(delta if s.turn==0 else -delta)
   if d['task']['split']!='train':ridge.append(rv(s))
  play(s,Move(*m) if m is not None else None,validate=False)
 assert s.done and [s.score(p) for p in [0,1]]==d['scores']
 return dict(id=d['task']['id'],group=d['task']['group'],trace=d['trace_sha256'],keys=keys,b=np.array(b,np.uint8),st=np.array(st,np.float32),ex=np.array(ex,np.float32),y=np.array(y,np.float32),ply=np.array(plies,np.int16),ridge=np.array(ridge,np.float32))

def dataset(name,tasks,forbidden=None):
 target=R/'features'/(name+'.npz');meta=target.with_suffix('.json')
 if target.exists():
  saved=read(meta);assert saved['tasks']==[t['id'] for t in tasks]
  return {k:v for k,v in np.load(target).items()},saved
 target.parent.mkdir(exist_ok=True);forbidden=set(forbidden or []);kept=[];excluded=[];traces=set();allkeys=set();group_keys={};raws=[]
 with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
  for x in pool.map(features_one,[str(R/'games'/(t['id']+'.json.gz')) for t in tasks],chunksize=16):
   raws.append(x);group_keys.setdefault(x['group'],set()).update(x['keys'])
 bad={g for g,ks in group_keys.items() if ks&forbidden}
 for x in raws:
  if x['group'] in bad:excluded.append(dict(id=x['id'],reason='reserved-state-overlap'));continue
  if x['trace'] in traces:excluded.append(dict(id=x['id'],reason='duplicate-trajectory'));continue
  traces.add(x['trace']);allkeys.update(x['keys']);kept.append(x)
 if not kept:raise RuntimeError('empty dataset '+name)
 arrays={k:np.concatenate([x[k] for x in kept]) for k in ['b','st','ex','y','ply','ridge']}
 arrays['w']=np.concatenate([np.full(len(x['y']),1/len(x['y']),np.float32) for x in kept]);arrays['game']=np.concatenate([np.full(len(x['y']),i,np.int32) for i,x in enumerate(kept)])
 arrays['group']=np.concatenate([np.full(len(x['y']),x['group']) for x in kept])
 tmp=target.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(target)
 metadata=dict(name=name,tasks=[t['id'] for t in tasks],unique_games=len(kept),positions=len(arrays['y']),groups=len({x['group'] for x in kept}),excluded=excluded,keys=sorted(allkeys),traces=sorted(traces),game_ids=[x['id'] for x in kept],sha256=sha(target))
 write(meta,metadata);event('features-ready',dataset=name,unique_games=len(kept),positions=len(arrays['y']),excluded=len(excluded));return arrays,metadata

def old_keys():
 keys=set(read(OLD/'old-state-hashes.json.gz'))
 for p in (OLD/'features').glob('*.json'):keys.update(read(p)['keys'])
 return keys

def reserve_prefixes():
 path=R/'reserved-prefix-keys.json'
 if path.exists():return set(read(path))
 keys=set();identities={}
 for t in read(R/'tasks.json'):
  if t['split'] not in ('benchmark','final','standard'):continue
  k=(t['split'],t['index'])
  if str(k) in identities:continue
  s,m=prefix(t,pair(SEEDS[0])[0]);h=state_hash(s);keys.add(h);identities[str(k)]=dict(hash=h,moves=m)
 write(path,sorted(keys));write(R/'frozen-prefixes.json',identities);return keys

def metrics(y,p,w):
 e=p-y;return dict(mae=float(np.average(abs(e),weights=w)),bias=float(np.average(e,weights=w)),rmse=float(np.sqrt(np.average(e*e,weights=w))))
def predict(model,data,device):
 model.eval();out=[]
 with torch.inference_mode():
  for i in range(0,len(data['y']),1024):out.append(model(*[torch.as_tensor(data[k][i:i+1024],device=device) for k in ['b','st','ex']]).cpu().numpy())
 return np.concatenate(out)
def details(data,p):
 r=metrics(data['y'],p,data['w']);r['stages']={name:metrics(data['y'][j],p[j],data['w'][j]) for name,j in [('early',data['ply']<16),('middle',(data['ply']>=16)&(data['ply']<30)),('late',data['ply']>=30)] if np.any(j)};return r

def cap():
 if time.time()-read(R/'plan.json')['created_at']>86400:raise RuntimeError('24-hour operational limit reached; checkpoints retained')
 if Path('/proc/meminfo').exists():
  kb=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
  if kb<4*1024*1024:raise RuntimeError('less than 4GiB available RAM; preserve production services')

def train(g,seed,tr,va,base):
 path=R/f'models/g{g}-{seed}.pt';last=R/f'models/g{g}-{seed}-last.pt';result=R/f'training/g{g}-{seed}.json'
 if result.exists():
  r=read(result);assert sha(path)==r['sha256'];return r
 torch.manual_seed(seed+g*100);np.random.seed(seed+g*100);device='cuda';torch.cuda.set_per_process_memory_fraction(.65)
 model=MarginNet().to(device);model.load_state_dict(torch.load(base,map_location=device,weights_only=False)['model'])
 opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=2,factor=.5,min_lr=1e-5)
 rng=np.random.default_rng(seed+g*100);history=[];epoch0=0;updates=0;seconds=0.;best=float('inf');best_epoch=-1;stale=0
 if last.exists():
  ck=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu());history=ck['history'];epoch0=ck['epoch']+1;updates=ck['updates'];seconds=ck['seconds'];best=ck['best'];best_epoch=ck['best_epoch'];stale=ck['stale']
 else:
  initial=details(va,predict(model,va,device));best=initial['mae'];history=[dict(epoch=-1,validation=initial,seconds=0,updates=0)]
  save(path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},komi=4,seed=seed,generation=g,epoch=-1,parent_sha256=sha(base)))
 # Load feature arrays once, not from compressed NPZ for every minibatch.
 n=len(tr['y']);weights=tr['w']*n/sum(tr['w'])
 for ep in range(epoch0,50):
  cap();start=time.monotonic();model.train();order=rng.permutation(n);loss_sum=0.;nb=0
  for i in range(0,n,512):
   j=order[i:i+512];args=[torch.as_tensor(tr[k][j],device=device) for k in ['b','st','ex']];y=torch.as_tensor(tr['y'][j],device=device);w=torch.as_tensor(weights[j],device=device)
   opt.zero_grad();p=model(*args);loss=(torch.nn.functional.smooth_l1_loss(p/32,y/32,reduction='none')*w).mean()
   if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();updates+=1;nb+=1;loss_sum+=loss.item()
  val=details(va,predict(model,va,device));scheduler.step(val['mae']);seconds+=time.monotonic()-start
  improved=val['mae']<best-1e-4
  if improved:best=val['mae'];best_epoch=ep;stale=0;save(path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},komi=4,seed=seed,generation=g,epoch=ep,parent_sha256=sha(base)))
  else:stale+=1
  history.append(dict(epoch=ep,train_loss=loss_sum/nb,validation=val,seconds=seconds,updates=updates,lr=opt.param_groups[0]['lr']))
  save(last,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},komi=4,seed=seed,generation=g,epoch=ep,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),history=history,updates=updates,seconds=seconds,best=best,best_epoch=best_epoch,stale=stale))
  event('training',generation=g,seed=seed,epoch=ep+1,updates=updates,training_seconds=seconds,positions=n,validation=val,best_mae=best,selected_epoch=best_epoch)
  if ep+1>=8 and stale>=6:break
 r=dict(generation=g,seed=seed,epochs=len(history)-1,updates=updates,seconds=seconds,history=history,selected_epoch=best_epoch,validation_mae=best,sha256=sha(path),checkpoint=str(path),train_positions=n,selection='validation only; retained parent allowed')
 write(result,r);return r

def pack_training(current,previous):
 # New on-policy data dominates; previous new generations retain coverage.
 if not previous:return {k:current[k] for k in ['b','st','ex','y','w']}
 arrays={k:np.concatenate([x[k] for x in previous+[current]]) for k in ['b','st','ex','y']}
 oldmass=sum(sum(x['w']) for x in previous)
 arrays['w']=np.concatenate([x['w']/oldmass*.5 for x in previous]+[current['w']/sum(current['w'])*.5])
 return arrays

def snapshot(g,unique,positions):
 training=[read(p) for p in sorted((R/'training').glob('*.json'))]
 write(R/'learning-progress.json',dict(generation=g,additional_unique_train_games=unique,cumulative_unique_train_games=11000+unique,new_training_positions=positions,training=training))

def run_campaign():
 tasks=read(R/'tasks.json');assert sha(R/'tasks.json')==read(R/'plan.json')['tasks_sha256']
 for seed in SEEDS:assert sha(baseline(seed))==read(R/'plan.json')['baseline_models'][str(seed)]
 cap();base={str(s):baseline(s) for s in SEEDS};reserved=old_keys()|reserve_prefixes()
 run_games([t for t in tasks if t['split'] in ('validation','test')],base,'reserved-data')
 va,vm=dataset('validation',[t for t in tasks if t['split']=='validation'],reserved)
 te,tm=dataset('test',[t for t in tasks if t['split']=='test'],reserved|set(vm['keys']))
 reserved|=set(vm['keys'])|set(tm['keys'])
 if not (R/'baseline-validation.json').exists():write(R/'baseline-validation.json',{str(s):details(va,predict(NetworkEvaluator(baseline(s)).model,va,'cpu')) for s in SEEDS})
 run_games([t for t in tasks if t['split']=='benchmark' and t['generation']==5],base,'benchmark-5')
 # Exclude complete baseline benchmark trajectories, never use these as labels.
 for p in (R/'games').glob('benchmark-g5-*.json.gz'):reserved.update(features_one(p)['keys'])
 previous=[dict(np.load(OLD/f'features/train-g{g}.npz')) for g in range(1,6)]
 oldtraces=set().union(*(set(read(OLD/f'features/train-g{g}.json')['traces']) for g in range(1,6)))
 alltraces=set(oldtraces);allkeys=set();unique=0;positions=0
 for g in range(6,16):
  cap();models=generation_models(g);ts=[t for t in tasks if t['split']=='train' and t['generation']==g]
  run_games(ts,models,f'generation-{g}')
  tr,meta=dataset(f'train-g{g}',ts,reserved)
  if len(set(meta['traces'])&alltraces):raise RuntimeError('repeated training trajectory')
  if meta['unique_games']<2100:raise RuntimeError('too many excluded games; inspect before continuing')
  alltraces.update(meta['traces']);allkeys.update(meta['keys']);unique+=meta['unique_games'];positions+=meta['positions']
  merged=pack_training(tr,previous)
  for seed in SEEDS:train(g,seed,merged,va,models[str(seed)])
  previous.append(tr);snapshot(g,unique,positions);event('generation-trained',generation=g,additional_unique_train_games=unique,cumulative_unique_train_games=11000+unique)
  run_games([t for t in tasks if t['split']=='benchmark' and t['generation']==g],{str(s):str(R/f'models/g{g}-{s}.pt') for s in SEEDS},f'benchmark-{g}')
  # Protect all observed benchmark trajectories from future training.
  for p in (R/'games').glob(f'benchmark-g{g}-*.json.gz'):
   ks=set(features_one(p)['keys'])
   if ks&allkeys:raise RuntimeError('benchmark trajectory intersects training: stop and audit')
   reserved.update(ks)
 if unique<21000:raise RuntimeError('insufficient additional unique training games')
 selection={};bv=read(R/'baseline-validation.json')
 for seed in SEEDS:
  candidates=[dict(generation=5,checkpoint=baseline(seed),sha256=sha(baseline(seed)),validation_mae=bv[str(seed)]['mae'])]+[read(R/f'training/g{g}-{seed}.json') for g in range(6,16)]
  selection[str(seed)]=min(candidates,key=lambda x:(x['validation_mae'],x['generation']))
 if not (R/'selection.json').exists():write(R/'selection.json',dict(at=time.time(),models=selection,criterion='new validation only, never benchmark',additional_unique_train_games=unique))
 else:assert read(R/'selection.json')['models']==selection
 if not (R/'prediction-test.json').exists():
  result={str(s):details(te,predict(NetworkEvaluator(selection[str(s)]['checkpoint']).model,te,'cpu')) for s in SEEDS}
  for s in SEEDS:result[f'baseline-{s}']=details(te,predict(NetworkEvaluator(baseline(s)).model,te,'cpu'))
  write(R/'prediction-test.json',dict(models=result,groups=tm['groups'],positions=tm['positions'],komi=4))
 run_games([t for t in tasks if t['split'] in ('final','standard')],{s:r['checkpoint'] for s,r in selection.items()},'final-evaluation')
 event('computation-complete',additional_unique_train_games=unique,cumulative_unique_train_games=11000+unique,generations=10)

if __name__=='__main__':
 torch.set_num_threads(1);parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);a=parser.parse_args()
 lock=(R/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  {'prepare':prepare,'run':run_campaign}[a.stage]()
 except Exception as error:
  event('failed',error=repr(error));raise
