import subprocess,json
from pathlib import Path
import torch
from scripts.bench_ppo_runtime import equal
from blokus.ppo_minibatch import R,write
for mb in (16,64):
 base='artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt'
 for name,until,resume in [('full',4104,base),('split',4100,base),('split',4104,str(R/f'pilot-{mb}-split/iteration-4100.pt'))]:
  out=R/f'pilot-{mb}-{name}'
  subprocess.run(['.venv/bin/python','-m','blokus.ppo_minibatch','--output',str(out),'--resume',resume,'--until',str(until),'--minibatch',str(mb),'--exclusions',str(R/'excluded-starts.json')],check=True)
 a=torch.load(R/f'pilot-{mb}-full/iteration-4104.pt',weights_only=False);b=torch.load(R/f'pilot-{mb}-split/iteration-4104.pt',weights_only=False)
 keys=['model','past','optimizer','rng']
 print(a.keys(),flush=True)
 # RNG fields depend on runtime checkpoint schema, include every RNG key.
 keys=['model','past','optimizer']+[k for k in a if 'rng' in k.lower()]
 assert all(equal(a[k],b[k]) for k in keys);assert a['minibatch']==b['minibatch']==mb
 write(R/f'resume-{mb}.json',dict(pass_=True,comparison='4096+8 vs 4096+4+separate-process-resume+4',past_swap_iterations=[4100,4104],keys=keys,minibatch=mb))
