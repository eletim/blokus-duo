"""Unchanged PPO continuation with bounded wall time and external replay history."""
import argparse,copy,gzip,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from . import terminal_ppo as p

def history(c,path):
 if 'history_chain' in c:
  games=[]
  for entry in c['history_chain']:
   q=Path(path).parent/entry['path'];assert p.sha(q)==entry['sha256'];part=json.load(gzip.open(q,'rt'));assert len(part)==entry['games'];games.extend(part)
  return games
 if 'games_gzip' in c:return json.loads(gzip.decompress(c['games_gzip']))
 q=Path(path).parent/c['history'];assert p.sha(q)==c['history_sha256'];return json.loads(gzip.decompress(q.read_bytes()))

def train(seed,resume,output,iterations=2048,cap=1200,batch=16,save_steps=(1024,2048),forbidden_prefixes=None):
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 if (out/'config.json').exists() or any(out.glob('iteration-*.pt')):raise FileExistsError('output already contains a run; resume into a new directory to preserve checkpoints')
 begin=time.perf_counter()
 p.seed_all(1400);model=p.TerminalNet().cuda();past=copy.deepcopy(model);opt=torch.optim.Adam(model.parameters(),lr=.0003)
 c=torch.load(resume,map_location='cpu',weights_only=False);assert c['seed']==seed
 model.load_state_dict(c['model']);past.load_state_dict(c['past']);opt.load_state_dict(c['optimizer']);step=c['iteration'];initial=step;logs=c['logs'];games=history(c,resume);elapsed=c['elapsed']
 chain=[]
 if 'history_chain' in c:
  chain=[dict(e,path=os.path.relpath(Path(resume).parent/e['path'],out)) for e in c['history_chain']]
 elif 'history' in c:chain=[dict(path=os.path.relpath(Path(resume).parent/c['history'],out),sha256=c['history_sha256'],games=len(games))]
 saved_count=len(games) if chain else 0
 forbidden=None
 if forbidden_prefixes:
  from .tournament import read_prefixes,prefix_state
  forbidden={p.state_hash(prefix_state(row)) for row in read_prefixes(forbidden_prefixes)['prefixes']}
 random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
 p.write(out/'config.json',dict(seed=seed,resume=str(resume),resume_sha256=p.sha(resume),iterations=iterations,initial_iteration=initial,additional_cap_seconds=cap,batch=batch,save_steps=list(save_steps),forbidden_prefixes=str(forbidden_prefixes),forbidden_sha256=p.sha(forbidden_prefixes) if forbidden_prefixes else None,torch=torch.__version__,source_hashes=p.source_hashes()))
 def save():
  nonlocal saved_count
  if len(games)>saved_count:
   hp=out/f'history-{saved_count}-{len(games)}.json.gz';hp.write_bytes(gzip.compress(json.dumps(games[saved_count:],separators=(',',':')).encode(),mtime=0));chain.append(dict(path=hp.name,sha256=p.sha(hp),games=len(games)-saved_count));saved_count=len(games)
  ck=dict(format='terminal-ppo-external-history-v1',model=model.state_dict(),past=past.state_dict(),optimizer=opt.state_dict(),iteration=step,seed=seed,elapsed=elapsed+time.perf_counter()-begin,logs=logs,history_chain=list(chain),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),source_hashes=p.source_hashes())
  path=out/f'iteration-{step}.pt';torch.save(ck,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)
 while step<iterations and time.perf_counter()-begin<cap:
  rec,gs,gen=p.collect(model,past,step,seed,batch,forbidden_starts=forbidden);up=p.update(model,opt,rec);step+=1;games.extend(gs)
  logs.append(dict(iteration=step,generation=gen,update=up,opponents={m:float(np.mean([np.sign(g['scores'][g['learner_seat']]-g['scores'][1-g['learner_seat']]) for g in gs if g['opponent']==m])) for m in ['current','past','heuristic']}))
  if step%4==0:past.load_state_dict(model.state_dict())
  if step in save_steps:save()
  if step%16==0:p.write(out/'curve.json',logs);print(seed,step,round(time.perf_counter()-begin,2),up,flush=True)
 if step not in save_steps:save()
 p.write(out/'curve.json',logs);p.write(out/'result.json',dict(seed=seed,initial_iteration=initial,iteration=step,games=len(games),additional_seconds=time.perf_counter()-begin,complete=step==iterations))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--resume',required=True);a.add_argument('--output',required=True);a.add_argument('--iterations',type=int,default=2048);a.add_argument('--cap',type=float,default=1200);a.add_argument('--batch',type=int,default=16);a.add_argument('--save-steps',type=int,nargs='+',default=[1024,2048]);a.add_argument('--forbidden-prefixes');train(**vars(a.parse_args()))
