"""Finite root-policy learning with exact RNG/optimizer continuation."""
import argparse,json,time,random,os
from pathlib import Path
import numpy as np
import torch
from .root_policy import RootPolicy,masked_loss
from .ai import seed_all
from .research import sha,source_hashes
from .value_experiment import write
ROOT=Path('artifacts/root-policy-distillation')

def metrics(model,data,split):
    ids=torch.where(data['split']==split)[0];correct=[];top5=[];loss=[];rank=[]
    model.eval()
    with torch.no_grad():
        for rows in ids.split(128):
            logits=model(data['x'][rows],data['state'][rows])+data['prior'][rows];mask=data['mask'][rows];target=data['target'][rows]
            logits=logits.masked_fill(~data['legal_mask'][rows],-1e9);correct.extend((logits.argmax(1)==target).cpu().tolist())
            top5.extend((logits.topk(5,1).indices==target[:,None]).any(1).cpu().tolist());rank.extend((logits>logits.gather(1,target[:,None])).sum(1).cpu().tolist())
            loss.append((float(masked_loss(logits,mask,target)),len(rows)))
    return dict(rows=len(ids),top1=float(np.mean(correct)),top5=float(np.mean(top5)),mean_rank=float(np.mean(rank))+1,nll=sum(x*n for x,n in loss)/max(1,len(ids)))

def save(path,model,opt,step,seed,data_hash,seconds):
    ck=dict(format='root-policy-v1',model=model.state_dict(),optimizer=opt.state_dict(),step=step,seed=seed,data_sha256=data_hash,training_seconds=seconds,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],source_hashes=source_hashes())
    torch.save(ck,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)

def train(output,seed=901,steps=3000,resume=None,device='cuda'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);seed_all(900)
    model=RootPolicy().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    raw=np.load(ROOT/'data.npz');data={k:torch.as_tensor(raw[k],device=device) for k in ['x','state','mask','target','split','prior','legal_mask']}
    data_hash=sha(ROOT/'data.npz');step=0;elapsed=0.;seed_all(seed)
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['data_sha256']==data_hash
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];elapsed=ck['training_seconds']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'])
        if ck['cuda_rng']:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    else:
        save(out/'untrained.pt',model,opt,0,seed,data_hash,0)
    write(out/'config.json',dict(seed=seed,shared_init_seed=900,steps=steps,device=device,lr=.001,batch_size=32,loss='masked teacher-choice cross entropy; no value regression',torch=torch.__version__,data_sha256=data_hash,resume=str(resume),gpu_training_cap_seconds=1800))
    train_ids=torch.where(data['split']==0)[0];logs=[];start=time.perf_counter()
    if not resume:logs.append(dict(step=0,dev=metrics(model,data,1)))
    model.train()
    while step<steps and elapsed+time.perf_counter()-start<1800:
        rows=train_ids[torch.randint(len(train_ids),(32,),device=device)]
        logits=model(data['x'][rows],data['state'][rows])+data['prior'][rows]
        loss=masked_loss(logits,data['mask'][rows],data['target'][rows]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();step+=1
        if step in (1000,3000):
            if device.startswith('cuda'):torch.cuda.synchronize()
            logs.append(dict(step=step,loss=float(loss),dev=metrics(model,data,1)));model.train()
            save(out/f'step-{step}.pt',model,opt,step,seed,data_hash,elapsed+time.perf_counter()-start)
    if device.startswith('cuda'):torch.cuda.synchronize()
    elapsed+=time.perf_counter()-start
    save(out/'latest.pt',model,opt,step,seed,data_hash,elapsed)
    final=metrics(model,data,1);logs.append(dict(step=step,dev=final,supervised_validation=metrics(model,data,2)))
    write(out/'training.json',dict(logs=logs,seconds=elapsed,steps=step,seed=seed));print(seed,step,elapsed,final,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=901);p.add_argument('--steps',type=int,default=3000);p.add_argument('--resume');p.add_argument('--device',default='cuda');a=p.parse_args();train(a.output,a.seed,a.steps,a.resume,a.device)
