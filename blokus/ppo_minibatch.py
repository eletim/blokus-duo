"""Finite matched-experience minibatch experiment. Old histories referenced, never copied."""
import argparse,copy,gzip,json,time,os
from pathlib import Path
import numpy as np
import torch
from . import ppo_runtime as r
from .ppo_gpu import update_fast
from .ppo_gpu_train import sha
R=Path('artifacts/ppo-minibatch')
SEEDS=[1751401,1751402,1751403]
def write(p,d):Path(p).write_text(json.dumps(d,indent=2))
def run(output,resume,until,minibatch,exclusions,cap=3600):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 r.seed_all(0) # deterministic flags/one CPU thread; restore subsequently restores all RNG
 model,past,opt,ck=r.restore(resume,'cuda');step=ck['iteration'];seed=ck['seed'];assert ck['channels']==4
 if 'minibatch' in ck:assert ck['minibatch']==minibatch
 else:assert step==4096 or str(output).find('pilot')>=0
 base=dict(path=os.path.relpath(resume,output),sha256=sha(resume),iteration=step)
 forbidden=set(json.loads(Path(exclusions).read_text())['states']);logs=[];start_step=step;start=time.perf_counter();hp=output/f'history-{step+1}-{until}.jsonl.gz'
 if hp.exists():raise FileExistsError(hp)
 with gzip.open(hp,'wt',compresslevel=1) as stream:
  while step<until and time.perf_counter()-start<cap:
   rec,games,tally=r.collect(model,past,step,seed,forbidden_starts=forbidden,backend='native')
   metric=update_fast(model,opt,rec,packed=True,minibatch=minibatch);step+=1
   if step%4==0:past.load_state_dict(model.state_dict())
   t=time.perf_counter();stream.write(json.dumps(dict(iteration=step,games=games),separators=(',',':'))+'\n')
   logs.append(dict(iteration=step,generation=tally,update=metric,history_write_seconds=time.perf_counter()-t,opponent_wdl={kind:float(np.mean([np.sign(g['scores'][g['learner_seat']]-g['scores'][1-g['learner_seat']]) for g in games if g['opponent']==kind])) for kind in ('current','past','heuristic')}))
   if step%64==0:print(minibatch,seed,step,round(time.perf_counter()-start,2),flush=True)
 path=output/f'iteration-{step}.pt';r.save(path,model,past,opt,step,seed,'native',True,logs)
 ck=torch.load(path,map_location='cpu',weights_only=False);ck.update(minibatch=minibatch,update_mode='fast',base_checkpoint=base,history_segment=dict(path=hp.name,sha256=sha(hp),first=start_step+1,last=step),exclusion_sha256=sha(exclusions));torch.save(ck,path);r.export(path,output/f'inference-{step}.pt')
 result=dict(seed=seed,minibatch=minibatch,start=start_step,iteration=step,games=(step-start_step)*16,wall=time.perf_counter()-start,complete=step==until,checkpoint=str(path),sha256=sha(path),base=base,logs=logs);write(output/f'result-{step}.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--resume',required=True);p.add_argument('--until',required=True,type=int);p.add_argument('--minibatch',type=int,required=True);p.add_argument('--exclusions',required=True);p.add_argument('--cap',type=float,default=3600);run(**vars(p.parse_args()))
