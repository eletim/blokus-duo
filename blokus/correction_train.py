"""Masked same-depth root correction, shared initialization, exact continuation."""
import argparse,json,time,random,os
from pathlib import Path
import numpy as np
import torch
from .ai import seed_all
from .correction import CorrectionNet,masked_loss

from .research import sha,source_hashes
from .value_experiment import write
from .correction_data import ROOT

def save(path,model,opt,step,seed,data_hash,seconds):
 ck=dict(format='root-correction-v1',model=model.state_dict(),optimizer=opt.state_dict(),step=step,seed=seed,data_sha256=data_hash,training_seconds=seconds,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),source_hashes=source_hashes())
 torch.save(ck,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)

def metrics(model,data,split,alpha=1.):
 ids=torch.where(data['split']==split)[0];model.eval();errors=[];zero=[];regrets=[];base_regrets=[];retained=[];random_errors=[];top_errors=[]
 with torch.no_grad():
  for rows in ids.split(32):
   pred=model(data['board'][rows],data['state'][rows],data['x'][rows]);mask=data['mask'][rows];target=data['target'][rows];initial=data['initial'][rows]
   errors.extend(((pred[mask]-target[mask])**2).cpu().tolist());zero.extend((target[mask]**2).cpu().tolist())
   for typ,dest in [(True,random_errors),(False,top_errors)]:
    m=mask & (data['random'][rows] if typ else ~data['random'][rows]);dest.extend(((pred[m]-target[m])**2).cpu().tolist())
   truth=(initial+target).masked_fill(~mask,-float('inf'));corrected=(initial+alpha*pred).masked_fill(~mask,-float('inf'));base=initial.masked_fill(~mask,-float('inf'));best=truth.max(1).values
   regret=best-truth.gather(1,corrected.argmax(1)[:,None]).squeeze(1);base_regret=best-truth.gather(1,base.argmax(1)[:,None]).squeeze(1)
   regrets.extend(regret.cpu().tolist());base_regrets.extend(base_regret.cpu().tolist());retained.extend((regret<4e-6).cpu().tolist())
 return dict(rows=len(ids),mse=float(np.mean(errors)),zero_mse=float(np.mean(zero)),regret=float(np.mean(regrets)),zero_regret=float(np.mean(base_regrets)),teacher_tie_retention=float(np.mean(retained)),random_mse=float(np.mean(random_errors)),top_mse=float(np.mean(top_errors)))

def train(output,seed=1001,steps=3000,resume=None,device='cuda'):
 out=Path(output);out.mkdir(parents=True,exist_ok=True);seed_all(1000);model=CorrectionNet().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
 raw=np.load(ROOT/'data.npz');data={k:torch.as_tensor(raw[k],device=device) for k in raw.files};h=sha(ROOT/'data.npz');step=0;elapsed=0;seed_all(seed)
 if resume:
  ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['data_sha256']==h;model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];elapsed=ck['training_seconds'];random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
 else:save(out/'untrained.pt',model,opt,0,seed,h,0)
 write(out/'config.json',dict(seed=seed,initial_seed=1000,steps=steps,lr=.001,batch=32,loss='masked Huber depth4-depth2 in root points',torch=torch.__version__,data_hash=h,shared='initial weights,data,game split; seed changes minibatch order',resume=str(resume)))
 ids=torch.where(data['split']==0)[0];logs=[];begin=time.perf_counter();model.train()
 while step<steps and elapsed+time.perf_counter()-begin<600:
  rows=ids[torch.randint(len(ids),(32,),device=device)];pred=model(data['board'][rows],data['state'][rows],data['x'][rows]);loss=masked_loss(pred,data['target'][rows],data['mask'][rows]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();step+=1
  if step in [1000,3000]:
   torch.cuda.synchronize();save(out/f'step-{step}.pt',model,opt,step,seed,h,elapsed+time.perf_counter()-begin)
   logs.append(dict(step=step,dev={str(a):metrics(model,data,1,a) for a in [.25,.5,1.]}));model.train()
 torch.cuda.synchronize();elapsed+=time.perf_counter()-begin;save(out/'latest.pt',model,opt,step,seed,h,elapsed);write(out/'metrics.json',logs);print(seed,step,elapsed,logs,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=1001);p.add_argument('--steps',type=int,default=3000);p.add_argument('--resume');a=p.parse_args();train(a.output,a.seed,a.steps,a.resume)
if __name__=='__main__':main()
