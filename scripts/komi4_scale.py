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
R=Path('artifacts/komi4-scale');PRIOR=Path('artifacts/komi4-campaign');SEEDS=[1751401,1751402,1751403]
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
def task(split,i,generation=0,opp=None,seat=0):
 return dict(id=f'{split}-g{generation}-{i:05}-{opp or "mix"}-{seat}',split=split,generation=generation,index=i,group=f'{split}-g{generation}-{i:05}',seed=SEEDS[i%3],opponent=opp or ['self','self','gen4','prior','ridge'][i%5],seat=seat,prefix_seed=526000000+{'train':1000000,'validation':2000000,'test':3000000,'comparison':4000000,'selection':5000000,'standard':6000000}[split]+generation*10000+i,prefix_plies=0 if split=='standard' else [8,10,12][i%3],opening='standard' if split=='standard' else 'uniform-legal' if i%4==0 else 'policy-top16-uniform',budget_ms=500 if split in ('comparison','standard','selection') else 150,max_depth=6 if split in ('comparison','standard','selection') else 4,width=4)
def prepare():
 if (R/'plan.json').exists():return
 tasks=[]
 for g in range(1,6):tasks += [task('train',i,g,seat=i%2) for i in range(2400)]
 for split in ['validation','test']:tasks += [task(split,i,opp=['gen4','prior','self','ridge'][i%4],seat=i%2) for i in range(256)]
 for i in range(128):
  for opp in ['ridge','prior','gen4']:
   for seat in [0,1]:tasks.append(task('comparison',i,opp=opp,seat=seat))
 for i in range(3):
  for opp in ['ridge','prior','gen4']:
   for seat in [0,1]:tasks.append(task('standard',i,opp=opp,seat=seat))
 for g in range(1,6):
  for i in range(32):
   for seat in [0,1]:tasks.append(task('selection',i,g,opp='gen4',seat=seat))
 write(R/'tasks.json',tasks)
 write(R/'plan.json',dict(created_at=time.time(),komi=4,generations=5,train_unique_target_per_generation=2200,minimum_unique_train_total=10000,train_reserve_per_generation=200,model_seeds=SEEDS,workers=8,reserved_cpus=[0,1,2,3,12,13,14,15],max_wall_seconds=86400,max_epochs=50,min_epochs=8,patience=6,batch_size=512,learning_rate=.0003,selection='per-seed best validation MAE, early stop; final generation per-seed by same validation only; test used once after freeze',training='new generations accumulated, full shuffled epochs, each game equal loss mass; old data <=10% examples, unique new games counted separately',generation_policy='half-point selfplay plus gen4/prior/ridge, depth4 cap150ms; policy top16/random legal diverse prefixes; same deterministic selfplay seat-swaps not generated',comparison='128 fresh prefixes x 2 seats x 3 opponents =768; plus 18 standard games. 500ms width4 depth6 all players; no test-driven retraining',split='fixed independent prefix seeds; remove whole validation/test group on overlap; training excludes reserved prefix states and known old validation/test positions; duplicate traces never independently counted',stop='24h, nonfinite, resource failure; resume with atomic game/checkpoint identity',tasks_sha256=sha(R/'tasks.json'),prior_models={str(s):sha(PRIOR/f'gen1-{s}.pt') for s in SEEDS}))
 event('plan-frozen',train_target=11000,final_comparisons=786)

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
 other={'self':value,'gen4':Gen4Komi(v),'ridge':rv,'prior':net(str(PRIOR/f"gen1-{t['seed']}.pt"))}[t['opponent']]
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
 if g==1:return {str(s):str(PRIOR/f'gen1-{s}.pt') for s in SEEDS}
 return {str(s):str(R/f'models/g{g-1}-{s}.pt') for s in SEEDS}
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

def benchmark():
 tasks=[t for t in read(R/'tasks.json') if t['split']=='train' and t['generation']==1 and t['index']<48]
 r=run_games(tasks,generation_models(1),'benchmark');event('benchmark-complete',**r)

def initial_data():
 tasks=read(R/'tasks.json');models=generation_models(1)
 run_games([t for t in tasks if t['split'] in ('validation','test')],models,'reserved-data')
 run_games([t for t in tasks if t['split']=='train' and t['generation']==1 and t['index']<2200],models,'generation-1')
 event('generation-ready',generation=1)

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
 path=R/'old-state-hashes.json.gz'
 if path.exists():return set(read(path))
 keys=set()
 rows=read('artifacts/margin-pilot/split-records.json')
 for r in rows:
  s=State()
  for ply,m in enumerate(r['moves']):
   if ply>=len(r['prefix']):keys.add(state_hash(s))
   play(s,Move(*m) if m is not None else None,validate=False)
 for p in (PRIOR/'games').glob('*.json'):
  d=read(p);s=State()
  for ply,m in enumerate(d['moves']):
   if ply>=len(d['task']['prefix']):keys.add(state_hash(s))
   play(s,Move(*m) if m is not None else None,validate=False)
 write(path,sorted(keys));return keys

def reserve_prefixes():
 path=R/'reserved-prefix-keys.json'
 if path.exists():return set(read(path))
 keys=set()
 for t in read(R/'tasks.json'):
  if t['split'] not in ('comparison','standard','selection'):continue
  s,m=prefix(t,pair(t['seed'])[0]);keys.add(state_hash(s))
 write(path,sorted(keys));return keys

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

def run_campaign():
 tasks=read(R/'tasks.json');assert sha(R/'tasks.json')==read(R/'plan.json')['tasks_sha256']
 cap();models=generation_models(1)
 run_games([t for t in tasks if t['split'] in ('validation','test')],models,'reserved-data')
 forbidden=old_keys()|reserve_prefixes()
 va,vm=dataset('validation',[t for t in tasks if t['split']=='validation'],forbidden)
 te,tm=dataset('test',[t for t in tasks if t['split']=='test'],forbidden|set(vm['keys']))
 reserved=forbidden|set(vm['keys'])|set(tm['keys']);previous=[];unique_total=0;global_traces=set();global_keys=set()
 for g in range(1,6):
  cap();models=generation_models(g);ts=[t for t in tasks if t['split']=='train' and t['generation']==g and t['index']<2200]
  run_games(ts,models,f'generation-{g}')
  tr,meta=dataset(f'train-g{g}',ts,reserved)
  overlaps=global_traces&set(meta['traces'])
  if overlaps:raise RuntimeError('duplicate trajectories across generations: do not inflate counts')
  if meta['unique_games']<2000:raise RuntimeError('insufficient unique usable games; inspect exclusions before continuing')
  global_traces.update(meta['traces']);global_keys.update(meta['keys']);unique_total+=meta['unique_games'];merged=pack_training(tr,previous)
  for seed in SEEDS:train(g,seed,merged,va,models[str(seed)])
  previous.append(tr);write(R/'learning-progress.json',dict(generation=g,unique_train_games=unique_total,unique_positions=len(global_keys),training_positions=sum(len(x['y']) for x in previous),excluded_duplicates=sum(len(read(R/f'features/train-g{j}.json')['excluded']) for j in range(1,g+1)),generated_train_games=g*2200,training=[read(p) for p in sorted((R/'training').glob('*.json'))]))
  event('generation-trained',generation=g,unique_train_games=unique_total,positions=sum(len(x['y']) for x in previous))
  run_games([t for t in tasks if t['split']=='selection' and t['generation']==g],{str(s):str(R/f'models/g{g}-{s}.pt') for s in SEEDS},f'selection-{g}')
 if unique_total<10000:raise RuntimeError('target 10000 unique training games not met')
 selection={}
 for seed in SEEDS:
  candidates=[read(R/f'training/g{g}-{seed}.json') for g in range(1,6)];selection[str(seed)]=min(candidates,key=lambda x:(x['validation_mae'],x['generation']))
 freeze=dict(at=time.time(),models=selection,criterion='validation only; no final test or comparison used',unique_train_games=unique_total)
 if (R/'selection.json').exists():assert read(R/'selection.json')['models']==selection
 else:write(R/'selection.json',freeze)
 event('selection-frozen',unique_train_games=unique_total,selected_generations={s:r['generation'] for s,r in selection.items()})
 if not (R/'prediction-test.json').exists():
  final={}
  for seed in SEEDS:
   model=NetworkEvaluator(selection[str(seed)]['checkpoint']).model;final[str(seed)]=details(te,predict(model,te,'cpu'))
  final['ridge']=details(te,te['ridge'])
  for seed in SEEDS:final[f'prior-{seed}']=details(te,predict(NetworkEvaluator(PRIOR/f'gen1-{seed}.pt').model,te,'cpu'))
  write(R/'prediction-test.json',dict(models=final,groups=tm['groups'],unique_games=tm['unique_games'],positions=tm['positions'],komi=4))
 run_games([t for t in tasks if t['split'] in ('comparison','standard')],{s:r['checkpoint'] for s,r in selection.items()},'final-comparison')
 event('computation-complete',unique_train_games=unique_total,generations=5)

if __name__=='__main__':
 torch.set_num_threads(1);parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','benchmark','initial-data','run']);a=parser.parse_args()
 lock=(R/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  {'prepare':prepare,'benchmark':benchmark,'initial-data':initial_data,'run':run_campaign}[a.stage]()
 except Exception as error:
  event('failed',error=repr(error));raise
