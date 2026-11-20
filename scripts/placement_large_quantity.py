"""Descriptive common-dev loss by dataset size; never a model-selection arm."""
import json
from pathlib import Path
import torch
import numpy as np
import placement_large_train as t
import placement_scale as s
from blokus.ppo_policy import PPOAgent
from blokus.placement_value import PlacementValue
R=t.R
def main():
 t.e.setup();manifest=json.loads((R/'data-manifest.json').read_text());x=[torch.from_numpy(a).to('cuda') for a in t.arrays(manifest['dev_files'])];rows=[]
 for seed in t.e.SEEDS:
  p=PPOAgent(seed,'cuda',backend='native')
  for n,root in [(3072,Path('artifacts/value-placement-scale/n3072')),(6144,Path('artifacts/value-placement-6144')),(24576,R)]:
   for v in 'AC':
    src=root/f'{v}-{seed}.pt' if n!=24576 else root/f'epoch-16-{v}-{seed}.pt';ck=torch.load(src,map_location='cpu',weights_only=True);assert ck['epochs']==16;m=PlacementValue(p.model,v).to('cuda');m.load_state_dict(ck['model']);rows.append(dict(train_games=n,seed=seed,variant=v,epochs=16,dev_mse=s.mse(m,x),source=str(src),sha=t.e.sha(src)))
 t.e.write(R/'quantity-dev-curve.json',dict(rows=rows,note='same new dev games, fixed epoch16; descriptive only, no strength selection; new generated data completed6ply vs old500ms and GPU rounding differ'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4))
 for v in 'AC':
  sizes=[3072,6144,24576];ys=np.array([[r['dev_mse'] for r in rows if r['variant']==v and r['train_games']==n] for n in sizes]);ax.plot(sizes,ys.mean(1),'o-',label=v);ax.fill_between(sizes,ys.min(1),ys.max(1),alpha=.15)
 ax.set_xlabel('Total training games (all models use the same matched data)');ax.set_ylabel('New-dev terminal WDL MSE');ax.set_title('Descriptive size curve, fixed 16 epochs\nGeneration differs; loss is not playing strength');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(R/'quantity-dev-curve.png',dpi=160)
if __name__=='__main__':main()
