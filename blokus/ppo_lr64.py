"""Finite learning-rate contrast; fixed generation stream and explicit optimizer LR. Old histories referenced, never copied."""
import argparse,copy,gzip,json,time,os
from pathlib import Path
import numpy as np
import torch
from . import ppo_runtime as r
from .ppo_gpu import update_fast
from .ppo_gpu_train import sha
R=Path('artifacts/ppo-lr64')
CONDITIONS={'A':(16,.0003),'B':(64,.0003),'C':(64,.0006)}
SEEDS=[1751401,1751402,1751403]
def write(p,d):Path(p).write_text(json.dumps(d,indent=2))
def run(output,resume,until,condition,exclusions,cap=3600):
 minibatch,lr=CONDITIONS[condition]
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 r.seed_all(0) # deterministic flags/one CPU thread; restore subsequently restores all RNG
 model,past,opt,ck=r.restore(resume,'cuda');step=ck['iteration'];seed=ck['seed'];assert ck['channels']==4
 series_seed=seed+20000
 if 'condition' in ck:
  assert ck['condition']==condition and ck['minibatch']==minibatch and ck['learning_rate']==lr and ck['series_seed']==series_seed
  assert all(g['lr']==lr for g in opt.param_groups), 'optimizer LR differs from saved metadata'
 else:assert step==4096, 'fork only from fixed latest 4096'
 for group in opt.param_groups:group['lr']=lr
 base=dict(path=os.path.relpath(resume,output),sha256=sha(resume),iteration=step)
 monitor=ck['logs'][-63:] if 'condition' in ck else []
 forbidden=set(json.loads(Path(exclusions).read_text())['states']);logs=[];start_step=step;start=time.perf_counter();hp=output/f'history-{step+1}-{until}.jsonl.gz'
 if hp.exists():raise FileExistsError(hp)
 with gzip.open(hp,'wt',compresslevel=1) as stream:
  while step<until and time.perf_counter()-start<cap:
   rec,games,tally=r.collect(model,past,step,series_seed,forbidden_starts=forbidden,backend='native')
   metric=update_fast(model,opt,rec,packed=True,minibatch=minibatch);step+=1
   if step%4==0:past.load_state_dict(model.state_dict())
   t=time.perf_counter();stream.write(json.dumps(dict(iteration=step,games=games),separators=(',',':'))+'\n')
   logs.append(dict(iteration=step,generation=tally,update=metric,history_write_seconds=time.perf_counter()-t,opponent_wdl={kind:float(np.mean([np.sign(g['scores'][g['learner_seat']]-g['scores'][1-g['learner_seat']]) for g in games if g['opponent']==kind])) for kind in ('current','past','heuristic')}))
   monitor.append(logs[-1]);monitor=monitor[-64:]
   if len(monitor)>=64 and np.mean([x['update']['entropy'] for x in monitor[-64:]])<.05:raise RuntimeError('collapse: rolling entropy below .05')
   if len(monitor)>=32 and np.mean([x['update']['approx_kl'] for x in monitor[-32:]])>.2:raise RuntimeError('collapse: rolling KL above .2')
   if step%64==0:print(condition,seed,step,round(time.perf_counter()-start,2),flush=True)
 path=output/f'iteration-{step}.pt';r.save(path,model,past,opt,step,seed,'native',True,logs)
 ck=torch.load(path,map_location='cpu',weights_only=False);ck.update(condition=condition,learning_rate=lr,series_seed=series_seed,minibatch=minibatch,update_mode='fast',base_checkpoint=base,history_segment=dict(path=hp.name,sha256=sha(hp),first=start_step+1,last=step),exclusion_sha256=sha(exclusions));torch.save(ck,path);r.export(path,output/f'inference-{step}.pt')
 result=dict(condition=condition,learning_rate=lr,series_seed=series_seed,seed=seed,minibatch=minibatch,start=start_step,iteration=step,games=(step-start_step)*16,wall=time.perf_counter()-start,complete=step==until,checkpoint=str(path),sha256=sha(path),base=base,logs=logs);write(output/f'result-{step}.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--resume',required=True);p.add_argument('--until',required=True,type=int);p.add_argument('--condition',choices=CONDITIONS,required=True);p.add_argument('--exclusions',required=True);p.add_argument('--cap',type=float,default=3600);run(**vars(p.parse_args()))
