"""Fixed-update learning curves; game clusters and update seeds are distinct."""
import argparse,json,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
from .correction import CorrectionNet,masked_loss
from .correction_train import save
from .mask_data import ROOT
from .mask_model import CandidateNet
from .research import sha,source_hashes
from .ai import seed_all
from .value_experiment import write

def per_game(model,data,ids):
 model.eval();games={}
 with torch.no_grad():
  for rows in ids.split(32):
   pred=model(data['board'][rows],data['state'][rows],data['x'][rows]);mask=data['mask'][rows];target=data['target'][rows];initial=data['initial'][rows]
   truth=(initial+target).masked_fill(~mask,-float('inf'));base=initial.masked_fill(~mask,-float('inf'));student=(initial+.5*pred).masked_fill(~mask,-float('inf'));best=truth.max(1).values
   mse=torch.where(mask,(pred-target).square(),0).sum(1)/mask.sum(1);zero=torch.where(mask,target.square(),0).sum(1)/mask.sum(1)
   regret=best-truth.gather(1,student.argmax(1)[:,None]).squeeze(1);zreg=best-truth.gather(1,base.argmax(1)[:,None]).squeeze(1)
   for g,values in zip(data['game'][rows].cpu().tolist(),torch.stack((mse,zero,regret,zreg),1).cpu().tolist()):games.setdefault(g,[]).append(values)
 return {str(g):np.mean(v,axis=0).tolist() for g,v in games.items()}

def summary(games):
 a=np.array(list(games.values()));return dict(zip(['mse','zero_mse','regret','zero_regret'],a.mean(0).tolist()))

def model_hash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
 return h.hexdigest()

def train(size,fold,seed,condition,steps=1000,resume=None,output=None,time_cap=120):
 out=Path(output) if output else ROOT/f'n{size}-f{fold}-s{seed}-{condition}';out.mkdir(parents=True,exist_ok=True);seed_all(1100);model=CandidateNet(condition).cuda();initial_hash=model_hash(model)
 wd=0;opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=wd);path=ROOT/'compact.npz';raw=np.load(path);data={k:torch.as_tensor(raw[k],device='cuda') for k in raw.files};h=sha(path)
 train_ids=torch.where(~data['bank'] & (data['fold']!=fold))[0];test_ids=torch.where(data['bank'] & ((data['fold']==fold) if fold>=0 else torch.ones_like(data['bank'])))[0];seed_all(seed);step=0;elapsed=0.;logs=[]
 if resume:
  ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['data_sha256']==h;model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];elapsed=ck['training_seconds'];random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
 config=dict(size=size,fold=fold,seed=seed,condition=condition,steps=steps,lr=.001,weight_decay=wd,alpha=.5,batch=32,initial_seed=1100,initial_hash=initial_hash,parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,data_sha256=h,source_hashes=source_hashes(),train_rows=len(train_ids),test_rows=len(test_ids),train_games=len(torch.unique(data['game'][train_ids])),test_games=len(torch.unique(data['game'][test_ids])),resume=str(resume));write(out/'config.json',config)
 begin=time.perf_counter();model.train()
 while step<steps and elapsed+time.perf_counter()-begin<time_cap:
  rows=train_ids[torch.randint(len(train_ids),(32,),device='cuda')];pred=model(data['board'][rows],data['state'][rows],data['x'][rows]);loss=masked_loss(pred,data['target'][rows],data['mask'][rows]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();step+=1
  if step in [1000,3000]:
   if (fold==0 and seed==1101) or fold==-1:save(out/f'step-{step}.pt',model,opt,step,seed,h,elapsed+time.perf_counter()-begin)
   test=per_game(model,data,test_ids);tr=per_game(model,data,train_ids);logs.append(dict(step=step,effective_epochs=step*32/len(train_ids),train=summary(tr),test=summary(test),test_games=test,model_sha256=model_hash(model)));model.train()
 torch.cuda.synchronize();elapsed+=time.perf_counter()-begin
 # Small explicit runs retain resume checkpoints; large grid retains representative only.
 if output:save(out/'latest.pt',model,opt,step,seed,h,elapsed)
 result=dict(config=config,step=step,seconds=elapsed,logs=logs);write(out/'result.json',result);print(size,fold,seed,condition,step,elapsed,logs[-1]['test'] if logs else '',flush=True);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--size',type=int,required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=1101);p.add_argument('--condition',choices=['summary','cells'],default='summary');p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume');p.add_argument('--output');a=p.parse_args();train(**vars(a))
if __name__=='__main__':main()
