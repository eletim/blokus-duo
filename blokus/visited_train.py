"""Fixed absolute loss; raw/live checkpoint selectors on the identical trajectory."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .ai import seed_all
from .research import source_hashes,sha
from .value_search import OutcomeNet,base_value
from .aligned_train import predict,candidate_values
from .ranking_train import differences
from .aligned_data import BASE
from .value_experiment import write


def load(path,device):
    d=np.load(path);g,k,r=d['state'].shape[:3]
    b=np.unpackbits(d['board'],axis=-1)[...,:1176].reshape(g,k,r,6,14,14)
    return tuple(torch.tensor(a,device=device) for a in [b.astype(np.float32),d['state'],d['returns'].mean(-1),d['mask'],d['terminal'],d['score']])


def live_values(p,x,terminal,score,alpha=.5):
    base=base_value(x.reshape(-1,49)).reshape(p.shape)
    return torch.where(terminal,score,alpha*p+(1-alpha)*base)


def decision_metrics(p,target,mask):
    v=candidate_values(p,mask);t=candidate_values(target,mask);ties=v>=v.max(-1,keepdim=True).values-1e-7
    expected=(t*ties).sum(-1)/ties.sum(-1);regret=t.max(-1).values-expected
    pd=differences(v);td=differences(t);significant=td.abs()>.025
    return dict(decision_regret_points=float(regret.mean()*40),pair_mse=float((pd-td).square().mean()),
                ordering_accuracy=float(((pd>0)==(td>0))[significant].float().mean()) if significant.any() else None,
                optimal_choice_rate=float((regret<1e-7).float().mean()),groups=len(p))


def metrics(model,data):
    b,x,y,m,terminal,score=data
    with torch.no_grad():p=torch.cat([predict(model,b[i:i+64],x[i:i+64]) for i in range(0,len(b),64)])
    return dict(raw=decision_metrics(p,y,m),live=decision_metrics(live_values(p,x,terminal,score),y,m))


def train(path,dev_path,output,seed=601,steps=3000,seconds=90,device='cuda',resume=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists() and not resume:raise ValueError('existing training')
    seed_all(seed);model=OutcomeNet('pooled').to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.001)
    step=0;best={k:(float('inf'),float('inf')) for k in ['raw','live']};previous=0;ck=None
    config=dict(seed=seed,train_path=str(path),train_sha256=sha(path),dev_path=str(dev_path),dev_sha256=sha(dev_path),initialize_sha256=sha(BASE),
                steps=steps,seconds=seconds,device=device,torch=torch.__version__,source_hashes=source_hashes(),loss='absolute smoothL1 beta .3',batch_groups=32,lr=.0002,
                selectors=['raw','live alpha .5 + exact terminal score'],tie_policy='mean regret over predicted ties within 1e-7')
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        if ck['config']['train_sha256']!=config['train_sha256'] or ck['config']['dev_sha256']!=config['dev_sha256']:raise ValueError('dataset changed')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];best=ck['best'];previous=ck['training_seconds']
    else:model.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=False)['model'])
    data=load(path,device);dev=load(dev_path,device);b,x,y,m,_,_=data
    if ck:
        torch.set_rng_state(ck['torch_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        if ck['cuda_rng']:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    write(out/'config.json',config);start=time.perf_counter();initial=step
    with (out/'train.jsonl').open('a') as log:
        while step-initial<steps and time.perf_counter()-start<seconds:
            ids=torch.randint(len(b),(32,),device=device);batch=b[ids];tr=int(torch.randint(4,(),device=device))
            if tr&1:batch=batch.transpose(-1,-2)
            if tr&2:batch=batch.flip((-1,-2))
            opt.zero_grad();pred=predict(model,batch,x[ids]);loss=torch.nn.functional.smooth_l1_loss(pred[m[ids]],y[ids][m[ids]],beta=.3)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();step+=1
            if step%400==0 or step-initial==steps:
                diag=metrics(model,dev);record=dict(step=step,loss=float(loss.detach()),seconds=previous+time.perf_counter()-start,**diag)
                log.write(json.dumps(record)+'\n');log.flush();print(seed,record,flush=True)
                for kind in ['raw','live']:
                    criterion=(diag[kind]['decision_regret_points'],diag[kind]['pair_mse'])
                    if criterion<best[kind]:
                        best[kind]=criterion;torch.save(dict(format=4,arch='pooled',model=model.state_dict(),step=step,config=config,selector=kind),out/f'{kind}.pt')
        if device.startswith('cuda'):torch.cuda.synchronize()
        diag=metrics(model,dev);elapsed=time.perf_counter()-start
        for kind in ['raw','live']:
            if not (out/f'{kind}.pt').exists():torch.save(dict(format=4,arch='pooled',model=model.state_dict(),step=step,config=config,selector=kind),out/f'{kind}.pt')
        torch.save(dict(format=4,arch='pooled',model=model.state_dict(),optimizer=opt.state_dict(),step=step,best=best,config=config,training_seconds=previous+elapsed,
             torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],python_rng=random.getstate(),numpy_rng=np.random.get_state()),out/'latest.pt')
    summary=dict(seed=seed,steps=step,invocation_steps=step-initial,invocation_seconds=elapsed,best=best,final_metrics=diag);write(out/'summary.json',summary);return summary
