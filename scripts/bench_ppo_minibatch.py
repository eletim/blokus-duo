import copy,json,time,statistics
from pathlib import Path
import torch
from blokus import ppo_runtime as r
from blokus.ppo_gpu import update_fast
from blokus.ppo_minibatch import R,write
r.seed_all(0)
base=Path('artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt');m,p,o,ck=r.restore(base,'cuda');records,games,t=r.collect(m,p,4096,1751401,backend='native');torch.save(records,R/'benchmark-rollout.pt');rows=[]
for rep in range(-2,6):
 for mb in ([16,64] if rep%2==0 else [64,16]):
  m,p,o,ck=r.restore(base,'cuda');r.seed_all(176001);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();result=update_fast(m,o,records,packed=True,minibatch=mb);b.record();b.synchronize();wall=time.perf_counter()-start
  # Replay the update's CPU permutation without modifying its saved RNG or training.
  with torch.random.fork_rng():
   torch.manual_seed(176001);ratios=[];remaining=result['updates']
   for _ in range(2):
    order=torch.randperm(len(records)).tolist()
    for i in range(0,len(order),mb):
     if remaining<=0:break
     ns=[len(records[j]['obs'][2]) for j in order[i:i+mb]];ratios.append((max(ns)*len(ns),sum(ns)));remaining-=1
  if rep>=0:rows.append(dict(rep=rep,minibatch=mb,wall=wall,cuda_span_ms=a.elapsed_time(b),peak_bytes=torch.cuda.max_memory_allocated(),padding_ratio=sum(x for x,y in ratios)/sum(y for x,y in ratios),states_per_second=len(records)/wall,steps_per_second=result['updates']/wall,update=result))
for row in rows:
 n=len(records);mb=row['minibatch'];batches=(n+mb-1)//mb;seen=sum(min(mb,n-(i%batches)*mb) for i in range(row['update']['updates']));row['optimizer_state_visits']=seen;row['optimizer_state_visits_per_second']=seen/row['wall']
write(R/'benchmark.json',dict(rows=rows,warmup_per_condition=2,repeats=6,same_saved_rollout=True,records=len(records),candidate_counts=[len(x['obs'][2]) for x in records],summary={str(mb):{k:statistics.median(x[k] for x in rows if x['minibatch']==mb) for k in ('wall','cuda_span_ms','peak_bytes','padding_ratio','states_per_second','steps_per_second')} for mb in (16,64)},note='CUDA event span includes host dispatch gaps; not kernel-only. Algorithm differs in minibatch, steps and KL stopping.'))
print(json.loads((R/'benchmark.json').read_text())['summary'])
for mb in (16,64):
 m,p,o,ck=r.restore(base,'cuda')
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:update_fast(m,o,records,packed=True,minibatch=mb)
 write(R/f'profile-{mb}.json',[dict(key=e.key,calls=e.count,cpu_us=e.self_cpu_time_total,device_us=e.self_device_time_total) for e in prof.key_averages()])
