"""Fixed-head policy distillation; separate gen4 value never receives gradients."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json,time,random
from pathlib import Path
import numpy as np
import torch
import placement_experiment as e
from candidate_data import R,SEEDS,save
from blokus.gen4 import load
from blokus.ppo_policy import collate

def dataset(phase,seed,p):
 rows=[]
 for path in sorted((R/phase).glob(f'{seed}-*.json')):
  g=json.loads(path.read_text())
  for r in g['rows']:
   s=e.state(g['moves'][:r['ply']]);obs,qs=p._observation(s);moves=[e.raw(q.move) for q in qs];ix=[moves.index(m) for m in r['roots']]
   assert len(ix)==len(set(ix)) and r['results']['wide']['depth']==6
   scores=torch.tensor(r['results']['wide']['root_scores']);target=torch.zeros(len(qs));target[ix]=torch.softmax(scores/.1,dim=0)
   with torch.inference_mode():l,v=p.model(*collate([obs],'cpu'))
   rows.append(dict(obs=obs,target=target,original=torch.log_softmax(l[0],dim=0),teacher=moves.index(r['results']['wide']['move']),stage=r['stage'],turn=r['turn'],hash=r['state_hash'],value=v.item()))
 assert rows
 for stage in range(3):assert sum(r['stage']==stage and r['turn']==0 for r in rows)==sum(r['stage']==stage and r['turn']==1 for r in rows)
 return rows

def objective(model,rows,device):
 inputs=collate([r['obs'] for r in rows],device);l,_=model(*inputs);total=[]
 for i,r in enumerate(rows):
  n=len(r['target']);lp=torch.log_softmax(l[i,:n],dim=0);target=r['target'].to(device);old=r['original'].to(device)
  total.append(-(target*lp).sum()+.5*(old.exp()*(old-lp)).sum())
 return torch.stack(total).mean()
def metrics(model,rows,device):
 model.eval();loss=0.;hit=0;top4=0;st={str(i):dict(n=0,top4=0,agreement=0) for i in range(3)}
 with torch.inference_mode():
  for start in range(0,len(rows),16):
   rs=rows[start:start+16];loss+=float(objective(model,rs,device))*len(rs);l,_=model(*collate([r['obs'] for r in rs],device))
   for j,r in enumerate(rs):
    order=l[j,:len(r['target'])].argsort(descending=True).tolist();a=int(order[0]==r['teacher']);t=int(r['teacher'] in order[:4]);hit+=a;top4+=t;z=st[str(r['stage'])];z['n']+=1;z['top4']+=t;z['agreement']+=a
 for z in st.values():
  if z['n']:z['top4']/=z['n'];z['agreement']/=z['n']
 return dict(loss=loss/len(rows),agreement=hit/len(rows),top4=top4/len(rows),stage=st,n=len(rows))

def train():
 assert json.loads((R/'diagnostic-gate.json').read_text())['passed'];torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available();start=time.perf_counter()
 for seed in SEEDS:
  p,value=load(seed,'native');original=copy.deepcopy(p.model.state_dict());tr=dataset('train',seed,p);dv=dataset('dev',seed,p)
  assert not {r['hash'] for r in tr}&{r['hash'] for r in dv}
  model=p.model.to('cuda');
  for name,param in model.named_parameters():param.requires_grad_(name.startswith('policy.'))
  opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4);rng=np.random.default_rng(seed+923);torch.manual_seed(seed);updates=0;curve=[]
  directory=R/f'models/{seed}';directory.mkdir(parents=True,exist_ok=True)
  for epoch in range(9):
   resumed=directory/f'epoch-{epoch}.pt'
   if resumed.exists():
    ck=torch.load(resumed,map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu());updates=ck['updates'];curve.append(ck['curve_row']);continue
   if epoch:
    model.train();order=rng.permutation(len(tr));sums=0
    for off in range(0,len(order),16):
     rows=[tr[i] for i in order[off:off+16]];opt.zero_grad();loss=objective(model,rows,'cuda');assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(norm);opt.step();updates+=1;sums+=float(loss)*len(rows)
   else:sums=metrics(model,tr,'cuda')['loss']*len(tr)
   dev=metrics(model,dv,'cuda');row=dict(epoch=epoch,updates=updates,train_loss=sums/len(tr),dev=dev);curve.append(row)
   frozen={k:t.cpu() for k,t in model.state_dict().items() if not k.startswith('policy.')};assert all(torch.equal(t,original[k]) for k,t in frozen.items())
   ck=dict(model={k:t.detach().cpu() for k,t in model.state_dict().items()},optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),updates=updates,epoch=epoch,curve_row=row)
   tmp=resumed.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(resumed);save(R/f'curve-{seed}.json',dict(seed=seed,train_labels=len(tr),dev_labels=len(dv),curve=curve));print('TRAIN',seed,epoch,updates,dev,flush=True)
  # Frozen network tensors are sufficient for exact value preservation on every input.
  assert any(not torch.equal(model.state_dict()[k].cpu(),original[k]) for k in original if k.startswith('policy.'))
  del model,p,opt;torch.cuda.empty_cache()
 curves=[json.loads((R/f'curve-{s}.json').read_text())['curve'] for s in SEEDS];means=[float(np.mean([c[i]['dev']['loss'] for c in curves])) for i in range(9)];chosen=min(range(9),key=means.__getitem__)
 save(R/'selection.json',dict(epoch=chosen,mean_dev_loss=means,rule='mean dev objective; tie earliest; epoch0 included',middle_regression=float(np.mean([c[chosen]['dev']['stage']['1']['top4']-c[0]['dev']['stage']['1']['top4'] for c in curves]))))
 for seed in SEEDS:
  ck=torch.load(R/f'models/{seed}/epoch-{chosen}.pt',map_location='cpu',weights_only=False);base=torch.load(f'artifacts/ppo-gen3/seed-{seed}.pt',map_location='cpu',weights_only=True);base['model']=ck['model'];base['candidate_policy']=dict(seed=seed,epoch=chosen,updates=ck['updates'],gen4_value_sha=e.sha(f'models/gen4/value-{seed}.pt'),method='root12-soft-distillation-head-only');torch.save(base,R/f'policy-{seed}.pt')
 save(R/'training-time.json',dict(seconds=time.perf_counter()-start,device='cuda',chosen_epoch=chosen))
if __name__=='__main__':train()
