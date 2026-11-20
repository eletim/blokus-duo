"""Common-new-dev descriptive loss only; no checkpoint or stage choice."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json
from pathlib import Path
import torch,numpy as np
import placement_large_train as t
import placement_scale as scale
from blokus.ppo_policy import PPOAgent
from blokus.placement_value import PlacementValue
R=Path('artifacts/value-placement-49152')
def main():
 t.e.setup();manifest=json.loads((R/'n49152/data-manifest.json').read_text());x=[torch.from_numpy(a) for a in t.arrays(manifest['dev_files'])];rows=[]
 for seed in t.e.SEEDS:
  p=PPOAgent(seed,'cpu',backend='native')
  for n,root in [(24576,Path('artifacts/value-placement-24576')),(36864,R/'n36864'),(49152,R/'n49152')]:
   for v in 'AC':
    for kind,src in [('selected',root/f'{v}-{seed}.pt'),('epoch16',root/f'epoch-16-{v}-{seed}.pt')]:
     ck=torch.load(src,map_location='cpu',weights_only=True);m=PlacementValue(p.model,v);m.load_state_dict(ck['model']);rows.append(dict(train_games=n,seed=seed,variant=v,kind=kind,epochs=ck['epochs'],updates=ck['updates'],dev_mse=scale.mse(m,x),source=str(src),sha=t.e.sha(src)))
 t.e.write(R/'quantity-dev.json',dict(rows=rows,note='same new192-game dev. New stages select on this dev, old24k selected on previous dev. Descriptive, not independent strength or a new selection. Fixed16 epoch and selected export shown separately; no extra training.'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,kind in zip(axes,('epoch16','selected')):
  for v in 'AC':
   ns=[24576,36864,49152];ys=np.array([[r['dev_mse'] for r in rows if r['kind']==kind and r['variant']==v and r['train_games']==n] for n in ns]);ax.plot(ns,ys.mean(1),'o-',label=v);ax.fill_between(ns,ys.min(1),ys.max(1),alpha=.15)
  ax.set(title=kind,xlabel='Cumulative training games',ylabel='Common-new-dev WDL MSE');ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Data-size loss curve (fixed three-seed range)\nNew dev is used for selection, not independent playing-strength evidence');fig.tight_layout();fig.savefig(R/'quantity-dev.png',dpi=160)
if __name__=='__main__':main()
