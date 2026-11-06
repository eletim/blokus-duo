import copy,json,time,statistics
from pathlib import Path
import torch,numpy as np
from blokus import ppo_runtime as r
from blokus.ppo_gpu import update_fast
R=Path('artifacts/ppo-ragged');R.mkdir(exist_ok=True)
def write(name,x):(R/name).write_text(json.dumps(x,indent=2)+'\n')
def main():
 r.seed_all(0);base='artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt'
 records=torch.load('artifacts/ppo-minibatch/benchmark-rollout.pt',weights_only=False);rows=[];checks=[]
 for mb in (16,64):
  weights={};metrics={}
  for rep in range(-2,6):
   for compact in ([False,True] if rep%2==0 else [True,False]):
    m,p,o,ck=r.restore(base,'cuda');r.seed_all(176001);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    result=update_fast(m,o,records,packed=not compact,minibatch=mb,compact=compact);wall=time.perf_counter()-start
    if rep>=0:rows.append(dict(rep=rep,minibatch=mb,compact=compact,wall=wall,peak_bytes=torch.cuda.max_memory_allocated(),update=result))
    weights[compact]={k:v.detach().clone() for k,v in m.state_dict().items()};metrics[compact]=result
  error=max((weights[True][k]-weights[False][k]).abs().max().item() for k in weights[True]);checks.append(dict(minibatch=mb,max_weight_error=error,steps=[metrics[c]['updates'] for c in (False,True)],metric_errors={k:abs(metrics[True][k]-metrics[False][k]) for k in ('loss','entropy','approx_kl','grad_norm')}))
 summary={f'{mb}-{c}':dict(wall=statistics.median(x['wall'] for x in rows if x['minibatch']==mb and x['compact']==c),peak_bytes=max(x['peak_bytes'] for x in rows if x['minibatch']==mb and x['compact']==c)) for mb in (16,64) for c in (False,True)}
 write('benchmark.json',dict(rows=rows,checks=checks,summary=summary,warmup=2,repeats=6,records=len(records)));print(summary,checks,flush=True)
if __name__=='__main__':main()
