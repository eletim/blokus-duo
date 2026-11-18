"""Descriptive held-out error slices; never used to select a model."""
import json
import numpy as np
import torch
import placement_scale as s
import placement_experiment as e
from blokus.engine import PIECES
from blokus.placement_value import PlacementValue
from blokus.ppo_policy import PPOAgent
R=s.R

def main():
 e.setup();device='cuda' if torch.cuda.is_available() else 'cpu';data=np.load(R/'dev-features.npz');labels=json.loads((R/'dev-labels.json').read_text())['labels'];x=[torch.tensor(data[f'x{i}'],device=device) for i in range(5)];inv=data['x1'][:,:21];available=data['x2'].reshape(-1,2,21,6)[:,0,:,0];areas=np.array([len(p) for p in PIECES]);blocked=(inv*(1-available)*areas).sum(1);ply=np.array([r['ply'] for r in labels]);masks=dict(all=np.ones(len(ply),bool),blocked_area_at_least_10=blocked>=10,blocked_area_under_10=blocked<10,ply_under20=ply<20,ply20to29=(ply>=20)&(ply<30),ply30plus=ply>=30);groups={}
 for name,mask in masks.items():groups[name]=dict(states=int(mask.sum()),games=len({(r['seed'],r['game']) for r,keep in zip(labels,mask) if keep}))
 scores={}
 for seed in e.SEEDS:
  p=PPOAgent(seed,device,backend='native')
  for name,size,variant in [('F',None,'C')]+[(f'{n}-{v}',n,v) for n in s.STAGES for v in 'ABC']:
   m=PlacementValue(p.model,variant).to(device)
   if size:
    ck=torch.load(R/f'n{size}'/f'{variant}-{seed}.pt',map_location=device,weights_only=True);m.load_state_dict(ck['model'])
   with torch.no_grad():pred=torch.cat([m(*(a[k:k+2048] for a in x[:3])) for k in range(0,len(ply),2048)]).cpu().numpy()
   err=(pred-data['x3'])**2;w=data['x4'];scores[f'{name}-{seed}']={g:float(np.sum(err[mask]*w[mask])/np.sum(w[mask])) if mask.any() else None for g,mask in masks.items()}
 e.write(R/'dev-diagnostics.json',dict(groups=groups,game_weighted_mse=scores,note='Descriptive correlated states from192heldoutgames, not model selection or causal proof. Blocked area=remaining own pieces with zero current legal placements, summed piece area.'))
 print('DEV DIAGNOSTICS COMPLETE',groups,flush=True)
if __name__=='__main__':main()
