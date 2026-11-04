"""Controlled loss ablations on shared root × opponent-reply groups."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .ai import seed_all
from .research import sha,source_hashes
from .value_search import OutcomeNet
from .value_experiment import write
from .ranking_train import differences
from .aligned_data import BASE


def load(path,device):
    d=np.load(path);g,k,r=d['state'].shape[:3]
    b=np.unpackbits(d['board'],axis=-1)[...,:1176].reshape(g,k,r,6,14,14)
    return (torch.tensor(b,dtype=torch.float32,device=device),torch.tensor(d['state'],device=device),torch.tensor(d['returns'].mean(-1),device=device),
            torch.tensor(d['mask'],device=device),torch.tensor(np.flatnonzero(d['game']%10!=0),device=device),torch.tensor(np.flatnonzero(d['game']%10==0),device=device))


def predict(model,b,x):
    shape=x.shape[:-1];return model(b.reshape(-1,6,14,14),x.reshape(-1,49)).reshape(shape)


def candidate_values(v,mask):return v.masked_fill(~mask,float('inf')).min(-1).values


def objective(p,y,mask,mode):
    absolute=torch.nn.functional.smooth_l1_loss(p[mask],y[mask],beta=.3)
    pd=differences(candidate_values(p,mask));td=differences(candidate_values(y,mask))
    diff=torch.nn.functional.smooth_l1_loss(pd,td,beta=.15)
    rank=torch.nn.functional.binary_cross_entropy_with_logits(pd/.15,torch.sigmoid(td/.15))
    if mode=='absolute':return absolute
    if mode=='difference':return absolute+.5*diff
    if mode=='ranking':return absolute+.1*rank
    raise ValueError('unknown objective')


def metrics(model,b,x,y,mask,ids):
    with torch.no_grad():p=torch.cat([predict(model,b[i],x[i]) for i in ids.split(64)])
    target=y[ids];m=mask[ids];v=candidate_values(p,m);t=candidate_values(target,m);choices=v.argmax(-1)
    regret=t.max(-1).values-t[torch.arange(len(t),device=t.device),choices]
    pd=differences(v);td=differences(t);significant=td.abs()>.025
    return dict(decision_regret_points=float(regret.mean()*40),optimal_choice_rate=float((regret<1e-7).float().mean()),
                pair_mse=float((pd-td).square().mean()),leaf_mse=float((p[m]-target[m]).square().mean()),
                ordering_accuracy=float(((pd>0)==(td>0))[significant].float().mean()),groups=len(ids))


def train(path,output,mode,seed=501,steps=6000,seconds=180,device='cuda',resume=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists() and not resume:raise ValueError('existing training')
    seed_all(seed);model=OutcomeNet('pooled').to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.001)
    step=0;best=(float('inf'),float('inf'));previous=0.;ck=None
    config=dict(mode=mode,seed=seed,dataset=str(path),dataset_sha256=sha(path),initialize=BASE,initialize_sha256=sha(BASE),steps=steps,seconds=seconds,
                device=device,torch=torch.__version__,source_hashes=source_hashes(),batch_groups=32,lr=.0002,augmentation='common transpose/180 rotation within group',checkpoint_selection='held-out-game decision regret; pair MSE breaks ties')
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        if ck['config']['dataset_sha256']!=config['dataset_sha256'] or ck['config']['mode']!=mode:raise ValueError('resume mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];best=ck['best'];previous=ck['training_seconds']
    else:model.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=False)['model'])
    b,x,y,m,ti,vi=load(path,device)
    if ck:
        torch.set_rng_state(ck['torch_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        if ck['cuda_rng']:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    write(out/'config.json',config);start=time.perf_counter();initial=step
    with (out/'train.jsonl').open('a') as log:
        while step-initial<steps and time.perf_counter()-start<seconds:
            ids=ti[torch.randint(len(ti),(32,),device=device)];batch=b[ids];tr=int(torch.randint(4,(),device=device))
            if tr&1:batch=batch.transpose(-1,-2)
            if tr&2:batch=batch.flip((-1,-2))
            opt.zero_grad();loss=objective(predict(model,batch,x[ids]),y[ids],m[ids],mode);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();step+=1
            if step%400==0 or step-initial==steps:
                diag=metrics(model,b,x,y,m,vi);record=dict(step=step,loss=float(loss.detach()),seconds=previous+time.perf_counter()-start,**diag)
                log.write(json.dumps(record)+'\n');log.flush();print(mode,seed,record,flush=True)
                criterion=(diag['decision_regret_points'],diag['pair_mse'])
                if criterion<best:
                    best=criterion;torch.save(dict(format=4,arch='pooled',model=model.state_dict(),step=step,config=config),out/'selected.pt')
        if device.startswith('cuda'):torch.cuda.synchronize()
        diag=metrics(model,b,x,y,m,vi);elapsed=time.perf_counter()-start
        if not (out/'selected.pt').exists():torch.save(dict(format=4,arch='pooled',model=model.state_dict(),step=step,config=config),out/'selected.pt')
        torch.save(dict(format=4,arch='pooled',model=model.state_dict(),optimizer=opt.state_dict(),step=step,best=best,config=config,training_seconds=previous+elapsed,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],python_rng=random.getstate(),numpy_rng=np.random.get_state()),out/'latest.pt')
    summary=dict(mode=mode,seed=seed,steps=step,invocation_steps=step-initial,invocation_seconds=elapsed,best=best,final_metrics=diag)
    write(out/'summary.json',summary);return summary
