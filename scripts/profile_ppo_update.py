"""Native collection and existing PPO update profiler; no algorithm changes."""
import copy,json,time
from pathlib import Path
import torch
from blokus import ppo_runtime as r
from blokus.ppo_policy import TerminalNet
root=Path('artifacts/ppo-four-gpu');root.mkdir(exist_ok=True,parents=True)
r.seed_all(1750000);m=TerminalNet(4).cuda();past=copy.deepcopy(m);opt=torch.optim.Adam(m.parameters(),lr=.0003)
rec,_,gen=r.collect(m,past,0,1750000,backend='native');r.update(m,opt,rec)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
 r.update(m,opt,rec)
prof.export_chrome_trace(str(root/'update-trace.json'))
(root/'update-profile.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=35)+'\n'+prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
rows=[]
for e in prof.key_averages():rows.append(dict(key=e.key,calls=e.count,cpu_us=e.self_cpu_time_total,device_us=e.self_device_time_total))
(root/'update-profile.json').write_text(json.dumps(dict(rows=rows,generation=gen,records=len(rec)),indent=2));print((root/'update-profile.txt').read_text()[:7000])
