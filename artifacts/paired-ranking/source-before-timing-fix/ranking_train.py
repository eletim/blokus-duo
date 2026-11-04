"""Within-root return differences and soft ordering, anchored to absolute returns."""
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


def differences(v):
    i,j=torch.triu_indices(v.shape[1],v.shape[1],1,device=v.device)
    return v[:,i]-v[:,j]


def losses(pred,target):
    pd=differences(pred);td=differences(target)
    diff=torch.nn.functional.smooth_l1_loss(pd,td,beta=.15)
    rank=torch.nn.functional.binary_cross_entropy_with_logits(pd/.15,torch.sigmoid(td/.15))
    absolute=torch.nn.functional.smooth_l1_loss(pred,target,beta=.3)
    return diff+.1*rank+.25*absolute,dict(diff=diff,rank=rank,absolute=absolute)


def dataset(path,device):
    d=np.load(path);g,k=d['state'].shape[:2]
    b=np.unpackbits(d['board'],axis=2)[:,:,:6*196].reshape(g,k,6,14,14)
    return (torch.tensor(b,dtype=torch.float32,device=device),torch.tensor(d['state'],device=device),
            torch.tensor(d['returns'].mean(2),device=device),torch.tensor(np.flatnonzero(d['game']%10!=0),device=device),
            torch.tensor(np.flatnonzero(d['game']%10==0),device=device))


def predict(model,b,x):
    g,k=x.shape[:2];return model(b.reshape(g*k,6,14,14),x.reshape(g*k,49)).reshape(g,k)


def metrics(model,b,x,y,ids):
    with torch.no_grad():p=torch.cat([predict(model,b[i],x[i]) for i in ids.split(128)])
    pd=differences(p);td=differences(y[ids]);mask=td.abs()>.025
    return dict(pair_mse=float((pd-td).square().mean()),absolute_mse=float((p-y[ids]).square().mean()),
                ordering_accuracy=float(((pd>0)==(td>0))[mask].float().mean()) if mask.any() else None,groups=len(ids))


def train(path,output,seed=301,steps=6000,seconds=180,device='cuda',resume=None,
          initialize='artifacts/value-search/compact/seed-201/predictive-best.pt'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists() and not resume:raise ValueError('checkpoint exists')
    seed_all(seed);model=OutcomeNet('pooled').to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.001)
    step=0;best=float('inf');previous=0.;ck=None
    config=dict(seed=seed,dataset=str(path),dataset_sha256=sha(path),initialize=str(initialize),initialize_sha256=sha(initialize),
                steps=steps,seconds=seconds,device=device,source_hashes=source_hashes(),torch=torch.__version__,batch_groups=64,
                objective='smoothL1 pair differences + .1 soft ranking BCE + .25 absolute smoothL1',lr=.0002,augment=True)
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        if ck['config']['dataset_sha256']!=config['dataset_sha256']:raise ValueError('dataset changed')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];best=ck['best'];previous=ck['training_seconds']
    else:model.load_state_dict(torch.load(initialize,map_location='cpu',weights_only=False)['model'])
    b,x,y,ti,vi=dataset(path,device)
    if ck:
        torch.set_rng_state(ck['torch_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        if ck['cuda_rng']:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    write(out/'config.json',config);start=time.perf_counter();initial=step
    with (out/'train.jsonl').open('a') as log:
        while step-initial<steps and time.perf_counter()-start<seconds:
            ids=ti[torch.randint(len(ti),(64,),device=device)];batch=b[ids]
            transform=int(torch.randint(4,(),device=device))
            if transform&1:batch=batch.transpose(-2,-1)
            if transform&2:batch=batch.flip((-2,-1))
            opt.zero_grad();loss,parts=losses(predict(model,batch,x[ids]),y[ids]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();step+=1
            if step%200==0 or step-initial==steps:
                m=metrics(model,b,x,y,vi);record=dict(step=step,loss=float(loss.detach()),seconds=previous+time.perf_counter()-start,**m)
                log.write(json.dumps(record)+'\n');log.flush();print(record,flush=True)
                if m['pair_mse']<best:
                    best=m['pair_mse'];torch.save(dict(format=4,arch='pooled',model=model.state_dict(),step=step,config=config),out/'predictive-best.pt')
        if device.startswith('cuda'):torch.cuda.synchronize()
        m=metrics(model,b,x,y,vi)
        if not (out/'predictive-best.pt').exists():
            best=m['pair_mse'];torch.save(dict(format=4,arch='pooled',model=model.state_dict(),step=step,config=config),out/'predictive-best.pt')
        elapsed=time.perf_counter()-start
        torch.save(dict(format=4,arch='pooled',model=model.state_dict(),optimizer=opt.state_dict(),step=step,best=best,config=config,
                        training_seconds=previous+elapsed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        python_rng=random.getstate(),numpy_rng=np.random.get_state()),out/'latest.pt')
    summary=dict(seed=seed,steps=step,invocation_steps=step-initial,invocation_seconds=elapsed,training_seconds=previous+elapsed,best_pair_mse=best,final_metrics=m)
    write(out/'summary.json',summary);return summary
