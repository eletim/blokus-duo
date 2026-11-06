import copy,json,time,statistics
from pathlib import Path
import torch
from blokus import ppo_runtime as r
from blokus.ppo_gpu import update_fast
from blokus.ppo_policy import TerminalNet
from scripts.bench_ppo_runtime import equal
root=Path('artifacts/ppo-four-gpu');r.seed_all(1750400);base=TerminalNet(4).cuda();rec,_,_=r.collect(base,copy.deepcopy(base),0,1750400,backend='native');rows=[];outs={}
for rep in range(-2,6):
 for mode in (['baseline','sync','sync-resident'] if rep%2==0 else ['sync-resident','sync','baseline']):
  m=copy.deepcopy(base);opt=torch.optim.Adam(m.parameters(),lr=.0003);r.seed_all(1750401);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  result=r.update(m,opt,rec) if mode=='baseline' else update_fast(m,opt,rec,packed=mode=='sync-resident');r.sync('cuda');wall=time.perf_counter()-start
  if rep>=0:
   rows.append(dict(rep=rep,mode=mode,wall=wall,update=result,memory=torch.cuda.max_memory_allocated()));outs[mode]=(copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state(),torch.cuda.get_rng_state_all())
 if rep>=0:assert equal(outs['baseline'],outs['sync']) and equal(outs['baseline'],outs['sync-resident'])
summary={mode:statistics.median(x['wall'] for x in rows if x['mode']==mode) for mode in ('baseline','sync','sync-resident')}
(root/'update-benchmark.json').write_text(json.dumps(dict(rows=rows,summary=summary,bit_equal=True,warmup_updates=6,compile=False),indent=2));print(summary)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:update_fast(m,opt,rec,packed=True)
(root/'fast-profile.json').write_text(json.dumps([dict(key=e.key,calls=e.count,cpu_us=e.self_cpu_time_total,device_us=e.self_device_time_total) for e in p.key_averages()],indent=2))
