"""Synchronous all-legal terminal PPO, optional native observations and resident rollout tensors.
No workers or background collection: snapshots occur at completed iteration boundaries.
"""
import argparse,copy,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .ppo_policy import TerminalNet,collate
from .ppo_native import observe as make_observation,scan,library
from .engine import State,Move,play,legal_moves,PLACEMENTS
from .ai import heuristic,seed_all

def state_hash(s):
 key=[s.boards,s.remaining,[s.score(0),s.score(1)],s.turn,s.passes]
 return hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()
def sync(device):
 if str(device).startswith('cuda'):torch.cuda.synchronize()
def legal(s,backend):
 if backend=='native' or (backend=='auto' and library() is not None):return [PLACEMENTS[i] for i in scan(s,False)[0]]
 return legal_moves(s)

def sample_batch(model,obs,tally=None):
 device=next(model.parameters()).device;t=time.perf_counter();batch=collate(obs,device);sync(device)
 if tally is not None:tally['transfer_pack_sync']+=time.perf_counter()-t
 t=time.perf_counter()
 with torch.no_grad():
  logits,values=model(*batch);dist=torch.distributions.Categorical(logits=logits);actions=dist.sample();lp=dist.log_prob(actions)
 result=(actions.cpu().tolist(),lp.cpu().tolist(),values.cpu().tolist())
 if tally is not None:tally['inference_and_output_sync']+=time.perf_counter()-t
 return result

def terminal_return(s,player):
 if not s.done:raise ValueError('incomplete game is not a terminal target')
 return float(np.sign(s.score(player)-s.score(1-player)))

def pool_modes(iteration,batch,kind="A"):
 assert kind in ("A","B") and batch%8==0
 modes=["current","current","past","heuristic"]*(batch//4)
 if kind=="B":
  for j,i in enumerate(range(2,batch,4)):modes[i]=("past","gen2","gen2048")[(iteration+j)%3]
 return modes

def collect(model,past,iteration,seed,batch=16,forbidden_starts=None,fixed_pool=None,backend="auto"):
 assert batch%8==0
 states=[State() for _ in range(batch)];hist=[[] for _ in states];traces=[[] for _ in states];records=[];modes=pool_modes(iteration,batch,'B' if fixed_pool else 'A');seats=[(i//4)%2 for i in range(batch)];tally=dict(cpu_generation_pack=0.,gpu_inference_sync=0.,other=0.,legal=0.,legal_and_native_features=0.,pack=0.,transfer_pack_sync=0.,inference_and_output_sync=0.,candidates=0);start=time.perf_counter();rngs=[];prefixes=[];prefix_rejections=[]
 for i,s in enumerate(states):
  rng=random.Random(seed*100000+(iteration*batch+i));rngs.append(rng);rejected=0
  while True:
   prefix=[];blocked=False
   for _ in range(rng.randint(2,6)):
    qs=legal(s,backend);m=rng.choice(qs).move if qs else None;play(s,m,validate=False);prefix.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
    if forbidden_starts and state_hash(s) in forbidden_starts:blocked=True
   if not blocked:break
   rejected+=1
   if rejected>100:raise RuntimeError('prefix rejection pool exhausted')
   s=State();states[i]=s
  prefix_rejections.append(rejected);prefixes.append(prefix)
 while not all(s.done for s in states):
  models={'current':model,'past':past,**(fixed_pool or {})};groups={k:[] for k in models};prepared={}
  for i,s in enumerate(states):
   if s.done:continue
   t=time.perf_counter();obs,qs=make_observation(s,model.channels,backend,tally);mode='current' if modes[i]=='current' or s.turn==seats[i] else modes[i]
   if mode=='heuristic':
    m=qs[int(np.argmax(heuristic(np.ascontiguousarray(obs[2][:,:12]))))].move if qs else None;hist[i].append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   else:
    prepared[i]=(obs,qs,s.turn,state_hash(s));groups[mode].append(i)
   tally['cpu_generation_pack']+=time.perf_counter()-t
  for mode,ids in groups.items():
   if not ids:continue
   t=time.perf_counter();actions,lp,vs=sample_batch(models[mode],[prepared[i][0] for i in ids],tally);tally['gpu_inference_sync']+=time.perf_counter()-t
   for j,i in enumerate(ids):
    obs,qs,player,h=prepared[i];a=actions[j];m=qs[a].move if qs else None
    if mode=='current':traces[i].append(dict(obs=obs,action=a,log_prob=lp[j],value=vs[j],player=player,state_hash=h,ply=len(prefixes[i])+len(hist[i])))
    hist[i].append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(states[i],m,validate=False)
 games=[]
 for i,s in enumerate(states):
  for r in traces[i]:r['return']=terminal_return(s,r['player']);r['opponent']=modes[i];records.append(r)
  games.append(dict(id=seed*100000+iteration*batch+i,opponent=modes[i],learner_seat=seats[i],prefix=prefixes[i],moves=hist[i],scores=[s.score(0),s.score(1)],targets=[{k:r[k] for k in ['ply','player','return','state_hash','action','log_prob','value']} for r in traces[i]]))
 if forbidden_starts is not None:
  for g,rejected in zip(games,prefix_rejections):g['prefix_rejections']=rejected
 tally['seconds']=time.perf_counter()-start;tally['other']=tally['seconds']-tally['cpu_generation_pack']-tally['gpu_inference_sync'];return records,games,tally

def update(model,opt,records,epochs=2,packed=False):
 start=time.perf_counter();device=next(model.parameters()).device;pack_start=time.perf_counter();resident=collate([r['obs'] for r in records],device) if packed else None;sync(device);pack_seconds=time.perf_counter()-pack_start
 ret=torch.tensor([r['return'] for r in records],device=device);oldv=torch.tensor([r['value'] for r in records],device=device);adv=ret-oldv;adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6);oldlp=torch.tensor([r['log_prob'] for r in records],device=device);actions=torch.tensor([r['action'] for r in records],device=device);logs=[];early=False
 for epoch in range(epochs):
  order=torch.randperm(len(records)).tolist()
  for offset in range(0,len(records),16):
   ix=order[offset:offset+16]
   if resident is None:batch=collate([records[i]['obs'] for i in ix],device)
   else:
    indices=torch.tensor(ix,device=device);n=max(len(records[i]['obs'][2]) for i in ix);batch=tuple(a.index_select(0,indices) if j<2 else a.index_select(0,indices)[:,:n].contiguous() for j,a in enumerate(resident))
   logits,v=model(*batch);dist=torch.distributions.Categorical(logits=logits);lp=dist.log_prob(actions[ix]);ratio=(lp-oldlp[ix]).exp();pl=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean();vl=(v-ret[ix]).square().mean();entropy=dist.entropy().mean();loss=pl+.5*vl-.01*entropy
   assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),.5);assert bool(torch.isfinite(norm));opt.step()
   with torch.no_grad():kl=((ratio-1)-(lp-oldlp[ix])).mean().item()
   logs.append([loss.item(),pl.item(),vl.item(),entropy.item(),kl,float(norm),float(((ratio-1).abs()>.2).float().mean())])
   if kl>.03:early=True;break
  if early:break
 sync(device);return dict(zip(['loss','policy_loss','value_mse','entropy','approx_kl','grad_norm','clip_fraction'],np.mean(logs,0).tolist()),updates=len(logs),records=len(records),early_kl=early,seconds=time.perf_counter()-start,rollout_pack_transfer_seconds=pack_seconds,packed=packed)

def save(path,model,past,opt,iteration,seed,backend,packed,logs,training_settings=None):
 device=next(model.parameters()).device;sync(device);t=time.perf_counter()
 ck=dict(format='terminal-ppo-runtime-v1',channels=model.channels,model=model.state_dict(),past=past.state_dict(),optimizer=opt.state_dict(),iteration=iteration,seed=seed,backend=backend,packed=packed,logs=logs,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],workers=None,checkpoint_boundary='completed iteration; no in-flight games')
 if training_settings is not None:ck.update(training_settings)
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,str(path)+'.tmp');os.replace(str(path)+'.tmp',path);return time.perf_counter()-t

def restore(path,device):
 ck=torch.load(path,map_location='cpu',weights_only=False)
 if ck.get('workers') is not None or ck.get('format')!='terminal-ppo-runtime-v1' or ck.get('channels') not in (4,6) or ck['model']['conv.0.weight'].shape[1]!=ck['channels']:raise ValueError('runtime checkpoint format/channels mismatch; no implicit legacy conversion')
 model=TerminalNet(ck['channels']).to(device);past=copy.deepcopy(model);opt=torch.optim.Adam(model.parameters(),lr=.0003);model.load_state_dict(ck['model']);past.load_state_dict(ck['past']);opt.load_state_dict(ck['optimizer'])
 random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);
 if ck['cuda_rng']:
  if not str(device).startswith('cuda'):raise ValueError('exact CUDA resume requires CUDA')
  torch.cuda.set_rng_state_all(ck['cuda_rng'])
 return model,past,opt,ck

def export(source,dest):
 ck=torch.load(source,map_location='cpu',weights_only=False)
 if ck.get('format')!='terminal-ppo-runtime-v1':raise ValueError('not runtime checkpoint')
 torch.save(dict(format='terminal-ppo-inference-v2',channels=ck['channels'],model=ck['model'],iteration=ck['iteration'],seed=ck['seed'],experimental=True),dest)

def train(output,iterations=64,seed=1741000,channels=4,device='cuda',backend='auto',packed=False,resume=None,cap=600,minibatch=None,learning_rate=None,allow_training_change=False):
 output=Path(output)
 if not resume and (output/'latest.pt').exists():raise FileExistsError('output already has a checkpoint; use --resume or a new directory')
 output.mkdir(parents=True,exist_ok=True);seed_all(seed)
 if resume:
  model,past,opt,ck=restore(resume,device);assert ck['channels']==channels and ck['seed']==seed and ck['backend']==backend and ck['packed']==packed;step=ck['iteration'];logs=ck['logs']
 else:
  model=TerminalNet(channels).to(device);past=copy.deepcopy(model);opt=torch.optim.Adam(model.parameters(),lr=.0003);step=0;logs=[];ck={}
 old_mb=ck.get('minibatch',16);old_lr=float(opt.param_groups[0]['lr'])
 if 'learning_rate' in ck and any(g['lr']!=ck['learning_rate'] for g in opt.param_groups):raise ValueError('checkpoint learning_rate metadata disagrees with optimizer')
 mb=old_mb if minibatch is None else minibatch;lr=old_lr if learning_rate is None else learning_rate
 if not isinstance(mb,int) or mb<=0 or not np.isfinite(lr) or lr<=0:raise ValueError('minibatch and learning-rate must be positive finite values')
 history=list(ck.get('training_changes',[]))
 if resume and (mb!=old_mb or lr!=old_lr):
  if not allow_training_change:raise ValueError('resume minibatch/LR change requires --allow-training-change; omitted values preserve checkpoint settings')
  history.append(dict(iteration=step,old_minibatch=old_mb,new_minibatch=mb,old_learning_rate=old_lr,new_learning_rate=lr,explicit=True))
 for group in opt.param_groups:group['lr']=lr
 settings=dict(minibatch=mb,learning_rate=lr,training_changes=history,update_implementation='consolidated-dense-v1')
 if not resume:save(output/'untrained.pt',model,past,opt,0,seed,backend,packed,logs,settings)
 start_step=step
 if (output/'games.jsonl').exists():
  lines=(output/'games.jsonl').read_text().splitlines()
  if lines and json.loads(lines[-1])['iteration']>step:raise ValueError('output game log is ahead of resume checkpoint')
 config=dict(seed=seed,channels=channels,iterations=iterations,backend=backend,packed=packed,device=device,batch_games=16,cap_seconds=cap,resume=str(resume) if resume is not None else None,torch=torch.__version__,loss='unchanged terminal WDL PPO',workers=0,compile=False,**settings);(output/'config.json').write_text(json.dumps(config,indent=2));start=time.perf_counter()
 while step<iterations and time.perf_counter()-start<cap:
  records,games,timing=collect(model,past,step,seed,backend=backend);from .ppo_gpu import update_fast
  metrics=update_fast(model,opt,records,packed=packed,minibatch=mb);step+=1
  if step%4==0:past.load_state_dict(model.state_dict())
  logs.append(dict(iteration=step,generation=timing,update=metrics,games_sha256=hashlib.sha256(json.dumps(games,sort_keys=True).encode()).hexdigest(),opponent_results={mode:float(np.mean([np.sign(g['scores'][g['learner_seat']]-g['scores'][1-g['learner_seat']]) for g in games if g['opponent']==mode])) for mode in ('current','past','heuristic')}))
  with (output/'games.jsonl').open('a') as stream:stream.write(json.dumps(dict(iteration=step,games=games))+'\n')
  print(step,round(time.perf_counter()-start,3),metrics,flush=True)
 seconds=time.perf_counter()-start;saving=save(output/'latest.pt',model,past,opt,step,seed,backend,packed,logs,settings);export(output/'latest.pt',output/'inference.pt');(output/'result.json').write_text(json.dumps(dict(complete=step==iterations,iteration=step,games=step*16,games_this_run=(step-start_step)*16,seconds_this_run=seconds,save_seconds=saving,logs=logs),indent=2));return model

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--iterations',type=int,default=64);p.add_argument('--seed',type=int,default=1741000);p.add_argument('--channels',type=int,choices=[4,6],default=4);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--backend',choices=['python','native','auto'],default='auto');p.add_argument('--packed',action='store_true');p.add_argument('--resume');p.add_argument('--cap',type=float,default=600);p.add_argument('--minibatch',type=int);p.add_argument('--learning-rate',type=float);p.add_argument('--allow-training-change',action='store_true');a=p.parse_args();train(**vars(a))
