import subprocess
import torch
from scripts.bench_ppo_runtime import equal
from blokus.ppo_lr64 import R,write,CONDITIONS
for condition,(mb,lr) in CONDITIONS.items():
 base='artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt'
 for name,until,resume in [('full',4104,base),('split',4100,base),('split',4104,str(R/f'pilot-{condition}-split/iteration-4100.pt'))]:
  subprocess.run(['.venv/bin/python','-m','blokus.ppo_lr64','--output',str(R/f'pilot-{condition}-{name}'),'--resume',resume,'--until',str(until),'--condition',condition,'--exclusions',str(R/'excluded-starts.json')],check=True)
 a=torch.load(R/f'pilot-{condition}-full/iteration-4104.pt',weights_only=False);b=torch.load(R/f'pilot-{condition}-split/iteration-4104.pt',weights_only=False);keys=['model','past','optimizer']+[k for k in a if 'rng' in k.lower()]
 assert all(equal(a[k],b[k]) for k in keys);assert a['learning_rate']==b['learning_rate']==lr and all(g['lr']==lr for g in a['optimizer']['param_groups'])
 write(R/f'resume-{condition}.json',dict(pass_=True,condition=condition,minibatch=mb,learning_rate=lr,comparison='4096+8 vs 4096+4+separate-process-resume+4',keys=keys,past_swaps=[4100,4104]))
