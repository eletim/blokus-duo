import copy,json
from pathlib import Path
import torch,numpy as np
from blokus import ppo_runtime as r
from blokus.ppo_policy import TerminalNet,collate
from blokus.ppo_ragged import CompactBatch,forward
from blokus.ppo_gpu import update_fast
R=Path('artifacts/ppo-ragged')
def check():
 rows=[];records=torch.load('artifacts/ppo-minibatch/benchmark-rollout.pt',weights_only=False)
 for device in ('cpu','cuda'):
  for channels in (4,6):
   r.seed_all(178100);m=TerminalNet(channels).to(device)
   if channels==4:m.load_state_dict(torch.load('artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt',weights_only=False)['model'])
   else:m.load_state_dict(torch.load('artifacts/ppo-gen2/seed-1401.pt',weights_only=True)['model'])
   rec=copy.deepcopy(records[:16])
   if channels==6:
    for x in rec:
     board=np.pad(x['obs'][0],((0,2),(0,0),(0,0)))
     for j,player in enumerate((x['player'],1-x['player'])):
      rc=(4,4) if player==0 else (9,9);board[4+j,rc[0],rc[1]]=1
     x['obs']=(board,*x['obs'][1:])
   a=copy.deepcopy(m);b=copy.deepcopy(m);obs=[x['obs'] for x in rec]
   la,va=a(*collate(obs,device));lb,vb=forward(b,CompactBatch(obs,device));finite=torch.isfinite(la);assert torch.equal(finite,torch.isfinite(lb))
   def loss(l,v):
    d=torch.distributions.Categorical(logits=l);actions=torch.tensor([x['action'] for x in rec],device=device);returns=torch.tensor([x['return'] for x in rec],device=device)
    return -d.log_prob(actions).mean()+.5*(v-returns).square().mean()-.01*d.entropy().mean()
   xa=loss(la,va);xb=loss(lb,vb);xa.backward();xb.backward()
   errors=dict(logit=(la[finite]-lb[finite]).abs().max().item(),value=(va-vb).abs().max().item(),loss=abs(xa.item()-xb.item()),grad=max((p.grad-q.grad).abs().max().item() for p,q in zip(a.parameters(),b.parameters())))
   oa=torch.optim.Adam(a.parameters(),lr=.0003);ob=torch.optim.Adam(b.parameters(),lr=.0003);oa.step();ob.step();errors['one_step_weight']=max((p-q).abs().max().item() for p,q in zip(a.parameters(),b.parameters()))
   rows.append(dict(device=device,channels=channels,errors=errors,within_preregistered_tolerance=all(x<=2e-5 for x in errors.values())))
 # Identify rollout-update discrepancy instead of hiding it behind matching metrics.
 for mb in (16,64):
  models=[]
  for compact in (False,True):
   r.seed_all(0);m,p,o,c=r.restore('artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt','cuda');r.seed_all(176001);result=update_fast(m,o,records,packed=not compact,minibatch=mb,compact=compact);models.append(m)
  errors={k:(models[0].state_dict()[k]-models[1].state_dict()[k]).abs().max().item() for k in models[0].state_dict()};rows.append(dict(minibatch=mb,weight_errors=errors))
 (R/'equivalence.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
if __name__=='__main__':check()
