import json,time,statistics
from pathlib import Path
import torch,numpy as np
from blokus import ppo_runtime as r
from blokus.ppo_gpu import update_fast
R=Path('artifacts/ppo-ragged');BASE='artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt'
def write(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
def main():
 records=torch.load('artifacts/ppo-minibatch/benchmark-rollout.pt',weights_only=False);rows=[]
 for phase,rec in [('early',[x for x in records if x['ply']<12]),('middle',[x for x in records if 12<=x['ply']<24]),('late',[x for x in records if x['ply']>=24])]:
  for mb in (16,64):
   for rep in range(-1,4):
    for compact in ([False,True] if rep%2==0 else [True,False]):
     r.seed_all(0);m,p,o,ck=r.restore(BASE,'cuda');r.seed_all(176001);torch.cuda.reset_peak_memory_stats();t=time.perf_counter();u=update_fast(m,o,rec,packed=not compact,minibatch=mb,compact=compact);wall=time.perf_counter()-t
     if rep>=0:rows.append(dict(phase=phase,records=len(rec),minibatch=mb,compact=compact,rep=rep,wall=wall,peak_bytes=torch.cuda.max_memory_allocated(),updates=u['updates'],states_per_second=len(rec)/wall))
 write('phase-benchmark.json',rows)
 for compact in (False,True):
  r.seed_all(0);m,p,o,ck=r.restore(BASE,'cuda');r.seed_all(176001)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:update_fast(m,o,records,packed=not compact,minibatch=64,compact=compact)
  write(f'profile-{compact}.json',[dict(key=e.key,calls=e.count,cpu_us=e.self_cpu_time_total,device_us=e.self_device_time_total) for e in prof.key_averages()])
 # Exact executed minibatch accounting for the benchmark permutation and early KL.
 data=json.loads((R/'benchmark.json').read_text())
 for row in data['rows']:
  r.seed_all(176001);remaining=row['update']['updates'];padded=actual=visits=0
  for epoch in range(2):
   order=torch.randperm(len(records)).tolist()
   for off in range(0,len(order),row['minibatch']):
    if remaining==0:break
    ns=[len(records[i]['obs'][2]) for i in order[off:off+row['minibatch']]];padded+=max(ns)*len(ns);actual+=sum(ns);visits+=len(ns);remaining-=1
  row.update(actual_candidates=actual,dense_candidates=padded,policy_candidates=actual if row['compact'] else padded,padding_ratio=padded/actual,optimizer_state_visits=visits,optimizer_states_per_second=visits/row['wall'])
 write('benchmark.json',data)
 print('profiles and phase benchmarks saved')
if __name__=='__main__':main()
