"""Same PPO update with consolidated host synchronization; reference remains ppo_runtime.update."""
import time
import numpy as np
import torch
from torch import nn
from .ppo_policy import collate
from .ppo_runtime import sync

def update_fast(model,opt,records,epochs=2,packed=False,minibatch=16):
 if not isinstance(minibatch,int) or minibatch<=0:raise ValueError("minibatch must be a positive integer")
 start=time.perf_counter();device=next(model.parameters()).device;pack_start=time.perf_counter();resident=collate([r['obs'] for r in records],device) if packed else None;sync(device);pack_seconds=time.perf_counter()-pack_start
 ret=torch.tensor([r['return'] for r in records],device=device);oldv=torch.tensor([r['value'] for r in records],device=device);adv=ret-oldv;adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6);oldlp=torch.tensor([r['log_prob'] for r in records],device=device);actions=torch.tensor([r['action'] for r in records],device=device);logs=[];early=False
 for epoch in range(epochs):
  order=torch.randperm(len(records)).tolist()
  for offset in range(0,len(records),minibatch):
   ix=order[offset:offset+minibatch]
   if resident is None:batch=collate([records[i]['obs'] for i in ix],device)
   else:
    indices=torch.tensor(ix,device=device);n=max(len(records[i]['obs'][2]) for i in ix);batch=tuple(a.index_select(0,indices) if j<2 else a.index_select(0,indices)[:,:n].contiguous() for j,a in enumerate(resident))
   logits,v=model(*batch);dist=torch.distributions.Categorical(logits=logits,validate_args=False);lp=dist.log_prob(actions[ix]);ratio=(lp-oldlp[ix]).exp();pl=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean();vl=(v-ret[ix]).square().mean();entropy=dist.entropy().mean();loss=pl+.5*vl-.01*entropy
   opt.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),.5)
   # One small D2H copy: validate before optimizer mutation and retain exact stepwise KL stop.
   with torch.no_grad():
    kl_tensor=((ratio-1)-(lp-oldlp[ix])).mean()
    row=torch.stack((loss,pl,vl,entropy,kl_tensor,norm,((ratio-1).abs()>.2).float().mean())).detach().cpu().tolist()
   if not np.isfinite(row).all():raise FloatingPointError('nonfinite PPO metrics/gradient; optimizer not stepped')
   opt.step();logs.append(row);kl=row[4]
   if kl>.03:early=True;break
  if early:break
 sync(device);return dict(zip(['loss','policy_loss','value_mse','entropy','approx_kl','grad_norm','clip_fraction'],np.mean(logs,0).tolist()),updates=len(logs),records=len(records),early_kl=early,seconds=time.perf_counter()-start,rollout_pack_transfer_seconds=pack_seconds,packed=packed)
