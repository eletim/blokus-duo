"""Batched CUDA outcome regression with exact optimizer/RNG continuation."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from .ai import seed_all
from .research import source_hashes,sha
from .value_search import OutcomeNet,base_value


def load_data(paths,device,branch_weight=1):
    datasets=[np.load(p) for p in paths]
    boards=np.concatenate([d['board'] for d in datasets]);state=np.concatenate([d['state'] for d in datasets])
    target=np.concatenate([d['target'] for d in datasets]);kind=np.concatenate([d['kind'] for d in datasets]);game=np.concatenate([d['game'] for d in datasets])
    board=np.unpackbits(boards,axis=1)[:,:6*196].reshape(-1,6,14,14)
    train=game%10!=0;valid=~train
    # Entire games held out for predictive diagnostics, never individual frames.
    indices=np.flatnonzero(train).tolist()+np.flatnonzero(train & (kind==1)).tolist()*(branch_weight-1)
    return (torch.tensor(board,dtype=torch.float32,device=device),torch.tensor(state,device=device),torch.tensor(target,device=device),
            torch.tensor(indices,device=device),torch.tensor(np.flatnonzero(valid),device=device))


def save(path,model,opt,step,config,seconds,best_mse):
    ck=dict(format=4,arch=model.arch,model=model.state_dict(),optimizer=opt.state_dict(),step=step,config=config,training_seconds=seconds,best_mse=best_mse,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    torch.save(ck,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)


def predictive_metrics(model,b,x,y,ids):
    with torch.no_grad():
        preds=[]
        for chunk in ids.split(1024):preds.append(model(b[chunk],x[chunk]))
        p=torch.cat(preds);target=y[ids];base=base_value(x[ids])
        return dict(mse=float((p-target).square().mean()),static_mse=float((base-target).square().mean()),
                    mae_points=float((p-target).abs().mean()*40),count=len(ids))


def train(paths,output,seed=1,steps=1600,seconds=120,device='cuda',resume=None,initialize=None,branch_weight=1,arch='flat',augment=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists() and resume is None:raise ValueError('already trained')
    seed_all(seed);model=OutcomeNet(arch).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.0001)
    step=0;previous=0.;best=float('inf')
    config=dict(seed=seed,paths=[str(p) for p in paths],dataset_hashes=[sha(p) for p in paths],steps=steps,seconds=seconds,device=device,
                branch_weight=branch_weight,arch=arch,augment=augment,initialize=str(initialize) if initialize else None,resume=str(resume) if resume else None,
                batch=256,lr=.0006,value_scale=40,torch=torch.__version__,cuda=torch.version.cuda,source_hashes=source_hashes())
    if initialize:
        model.load_state_dict(torch.load(initialize,map_location='cpu',weights_only=False)['model'])
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        if ck['config']['dataset_hashes']!=config['dataset_hashes'] or ck['config']['branch_weight']!=branch_weight or ck.get('arch','flat')!=arch or ck['config'].get('augment',False)!=augment:raise ValueError('resume dataset differs')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];previous=ck['training_seconds'];best=ck['best_mse']
    b,x,y,train_ids,val_ids=load_data(paths,device,branch_weight)
    if resume:
        torch.set_rng_state(ck['torch_rng'])
        if ck['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(ck['cuda_rng'])
    (out/'config.json').write_text(json.dumps(config,indent=2))
    start=time.perf_counter();initial_step=step
    with (out/'train.jsonl').open('a') as log:
        while step-initial_step<steps and time.perf_counter()-start<seconds:
            ids=train_ids[torch.randint(len(train_ids),(256,),device=device)]
            opt.zero_grad();batch=b[ids]
            if augment:
                transform=int(torch.randint(4,(),device=device))
                if transform&1:batch=batch.transpose(2,3)
                if transform&2:batch=batch.flip((2,3))
            pred=model(batch,x[ids]);loss=torch.nn.functional.smooth_l1_loss(pred,y[ids],beta=.3)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();step+=1
            if step%200==0 or step-initial_step==steps:
                metrics=predictive_metrics(model,b,x,y,val_ids)
                record=dict(step=step,loss=float(loss.detach()),seconds=previous+time.perf_counter()-start,**metrics)
                log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                if metrics['mse']<best:
                    best=metrics['mse']
                    # Small inference-only selected model; do not multiply optimizer checkpoints.
                    torch.save(dict(format=4,arch=model.arch,model=model.state_dict(),step=step,config=config),out/'predictive-best.pt')
        if device.startswith('cuda'):torch.cuda.synchronize()
        metrics=predictive_metrics(model,b,x,y,val_ids)
        if not (out/'predictive-best.pt').exists():torch.save(dict(format=4,arch=model.arch,model=model.state_dict(),step=step,config=config),out/'predictive-best.pt')
        elapsed=previous+time.perf_counter()-start;save(out/'latest.pt',model,opt,step,config,elapsed,best)
    summary=dict(seed=seed,steps=step,invocation_steps=step-initial_step,training_seconds=elapsed,invocation_seconds=time.perf_counter()-start,
                 best_mse=best if np.isfinite(best) else metrics['mse'],final_metrics=metrics,training_positions=len(train_ids),validation_positions=len(val_ids))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary
