"""Reference synchronous collect/PPO from a011c78, device and channel adapters only."""
import random,time
import numpy as np
import torch
from torch import nn
from .engine import State,legal_moves,play
from .ppo_policy import observe,collate
from .ai import features,heuristic
from .ppo_runtime import state_hash,terminal_return,pool_modes

def sample_batch(model,obs):
 with torch.no_grad():
  logits,values=model(*collate(obs,next(model.parameters()).device));dist=torch.distributions.Categorical(logits=logits);actions=dist.sample();lp=dist.log_prob(actions)
 return actions.cpu().tolist(),lp.cpu().tolist(),values.cpu().tolist()

def collect(model,past,iteration,seed,batch=16,forbidden_starts=None,fixed_pool=None):
 assert batch%8==0
 states=[State() for _ in range(batch)];hist=[[] for _ in states];traces=[[] for _ in states];records=[];modes=pool_modes(iteration,batch,'B' if fixed_pool else 'A');seats=[(i//4)%2 for i in range(batch)];tally=dict(cpu_generation_pack=0.,gpu_inference_sync=0.,other=0.);start=time.perf_counter();rngs=[];prefixes=[];prefix_rejections=[]
 for i,s in enumerate(states):
  rng=random.Random(seed*100000+(iteration*batch+i));rngs.append(rng);rejected=0
  while True:
   prefix=[];blocked=False
   for _ in range(rng.randint(2,6)):
    qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=False);prefix.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
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
   t=time.perf_counter();qs=legal_moves(s);mode='current' if modes[i]=='current' or s.turn==seats[i] else modes[i]
   if mode=='heuristic':
    m=qs[int(np.argmax(heuristic(features(s,qs)[0])))].move if qs else None;hist[i].append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   else:
    obs,_=observe(s,qs,model.channels);prepared[i]=(obs,qs,s.turn,state_hash(s));groups[mode].append(i)
   tally['cpu_generation_pack']+=time.perf_counter()-t
  for mode,ids in groups.items():
   if not ids:continue
   t=time.perf_counter();actions,lp,vs=sample_batch(models[mode],[prepared[i][0] for i in ids]);tally['gpu_inference_sync']+=time.perf_counter()-t
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

def update(model,opt,records,epochs=2):
 device=next(model.parameters()).device;start=time.perf_counter();ret=torch.tensor([r['return'] for r in records],device=device);oldv=torch.tensor([r['value'] for r in records],device=device);adv=ret-oldv;adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6);oldlp=torch.tensor([r['log_prob'] for r in records],device=device);actions=torch.tensor([r['action'] for r in records],device=device);logs=[];early=False
 for epoch in range(epochs):
  order=torch.randperm(len(records)).tolist()
  for offset in range(0,len(records),16):
   ix=order[offset:offset+16];logits,v=model(*collate([records[i]['obs'] for i in ix],device));dist=torch.distributions.Categorical(logits=logits);lp=dist.log_prob(actions[ix]);ratio=(lp-oldlp[ix]).exp();pl=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean();vl=(v-ret[ix]).square().mean();entropy=dist.entropy().mean();loss=pl+.5*vl-.01*entropy
   assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),.5);assert bool(torch.isfinite(norm));opt.step()
   with torch.no_grad():kl=((ratio-1)-(lp-oldlp[ix])).mean().item()
   logs.append([loss.item(),pl.item(),vl.item(),entropy.item(),kl,float(norm),float(((ratio-1).abs()>.2).float().mean())])
   if kl>.03:early=True;break
  if early:break
 if device.type=='cuda':torch.cuda.synchronize()
 return dict(zip(['loss','policy_loss','value_mse','entropy','approx_kl','grad_norm','clip_fraction'],np.mean(logs,0).tolist()),updates=len(logs),records=len(records),early_kl=early,seconds=time.perf_counter()-start)
