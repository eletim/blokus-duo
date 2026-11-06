"""Finite update-mode comparison and separate-process restart; no strength claims."""
import argparse,copy,json,time,hashlib
from pathlib import Path
import torch
from blokus import ppo_runtime as r
from blokus.ppo_gpu import update_fast
from scripts.bench_ppo_runtime import records_hash,equal
BASE='artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt'
def run(mode,until,out,resume=None,device='cuda'):
 r.seed_all(178200)
 if device=='cpu' and not resume:
  from blokus.ppo_policy import TerminalNet
  ck=torch.load(BASE,map_location='cpu',weights_only=False);m=TerminalNet(4);m.load_state_dict(ck['model']);p=copy.deepcopy(m);p.load_state_dict(ck['past']);o=torch.optim.Adam(m.parameters(),lr=.0003);o.load_state_dict(ck['optimizer'])
 else:m,p,o,ck=r.restore(resume or BASE,device)
 start=ck['iteration'];out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if resume:assert ck['update_mode']==mode
 logs=ck.get('pilot_logs',[]) if resume else []
 if not resume:r.seed_all(178200)
 begin=time.perf_counter()
 for i in range(start,until):
  rec,games,t=r.collect(m,p,i,178200,backend='native');h=records_hash(rec)
  if mode=='dense':u=update_fast(m,o,rec,packed=True,minibatch=64)
  else:u=update_fast(m,o,rec,compact=True,minibatch=64)
  if (i+1)%4==0:p.load_state_dict(m.state_dict())
  logs.append(dict(iteration=i+1,input_hash=h,games=games,timing=t,update=u))
 r.save(out/'latest.pt',m,p,o,until,178200,backend='native',packed=False,logs=[])
 x=torch.load(out/'latest.pt',weights_only=False);x['update_mode']=mode;x['pilot_logs']=logs;torch.save(x,out/'latest.pt')
 (out/'summary.json').write_text(json.dumps(dict(mode=mode,from_iteration=start,to_iteration=until,wall=time.perf_counter()-begin,logs=logs),indent=2)+'\n')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--mode',choices=['dense','compact'],required=True);a.add_argument('--until',type=int,required=True);a.add_argument('--out',required=True);a.add_argument('--resume');a.add_argument('--device',default='cuda');k=a.parse_args();run(k.mode,k.until,k.out,k.resume,k.device)
