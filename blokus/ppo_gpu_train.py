"""Finite four-channel training; native collection, identical PPO, external history segments."""
import argparse,copy,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from . import ppo_runtime as r
from .ppo_gpu import update_fast
from .ppo_policy import TerminalNet

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(output,seed,until,mode='fast',resume=None,cap=5400,exclusions=None,device='cuda'):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);r.seed_all(seed);history=[]
 if resume:
  model,past,opt,ck=r.restore(resume,device);assert ck['seed']==seed and ck['channels']==4
  step=ck['iteration'];logs=ck['logs'];assert ck.get('update_mode')==mode
  history=ck['history_chain']
  for entry in history:assert sha(output/entry['path'])==entry['sha256']
 else:
  if (output/'untrained.pt').exists():raise FileExistsError('use resume to retain completed work')
  model=TerminalNet(4).to(device);past=copy.deepcopy(model);opt=torch.optim.Adam(model.parameters(),lr=.0003);step=0;logs=[]
  r.save(output/'untrained.pt',model,past,opt,0,seed,'native',mode=='fast',logs);r.export(output/'untrained.pt',output/'inference-0.pt')
 forbidden=set(json.loads(Path(exclusions).read_text())['states']) if exclusions else set();start_step=step;start=time.perf_counter();hp=output/f'history-{step+1}-{until}.jsonl.gz'
 if hp.exists():raise FileExistsError(hp)
 with gzip.open(hp,'wt',compresslevel=1) as stream:
  while step<until and time.perf_counter()-start<cap:
   rec,games,tally=r.collect(model,past,step,seed,forbidden_starts=forbidden,backend='native')
   metric=update_fast(model,opt,rec,packed=True) if mode=='fast' else r.update(model,opt,rec)
   step+=1
   if step%4==0:past.load_state_dict(model.state_dict())
   t=time.perf_counter();stream.write(json.dumps(dict(iteration=step,games=games),separators=(',',':'))+'\n');storage=time.perf_counter()-t
   logs.append(dict(iteration=step,generation=tally,update=metric,history_write_seconds=storage,games_sha256=hashlib.sha256(json.dumps(games,sort_keys=True).encode()).hexdigest(),opponent_wdl={kind:float(np.mean([np.sign(g['scores'][g['learner_seat']]-g['scores'][1-g['learner_seat']]) for g in games if g['opponent']==kind])) for kind in ('current','past','heuristic')}))
   if step%64==0:print(seed,step,round(time.perf_counter()-start,2),metric['entropy'],metric['value_mse'],flush=True)
 history.append(dict(path=hp.name,sha256=sha(hp),first=start_step+1,last=step,games=(step-start_step)*16));path=output/f'iteration-{step}.pt'
 r.save(path,model,past,opt,step,seed,'native',mode=='fast',logs);ck=torch.load(path,map_location='cpu',weights_only=False);ck.update(update_mode=mode,history_chain=history,exclusion_sha256=sha(exclusions) if exclusions else None,source_hashes={str(p):sha(p) for p in (Path(__file__),Path(r.__file__),Path(__file__).with_name('ppo_gpu.py'))});torch.save(ck,path);r.export(path,output/f'inference-{step}.pt')
 result=dict(seed=seed,mode=mode,start=start_step,until=until,iteration=step,games_this_run=(step-start_step)*16,wall_with_save=time.perf_counter()-start,complete=step==until,checkpoint=str(path),checkpoint_sha256=sha(path),exclusion_count=len(forbidden),logs=logs[start_step:]);(output/f'result-{step}.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--until',type=int,required=True);p.add_argument('--mode',choices=['baseline','fast'],default='fast');p.add_argument('--resume');p.add_argument('--cap',type=float,default=5400);p.add_argument('--exclusions');p.add_argument('--device',default='cuda');run(**vars(p.parse_args()))
